import cv2
import numpy as np
from sam3p1 import (
    postprocess,
    preprocess,
    run_encoder,
    run_grounding,
    run_interactive_obj_ptr_proj,
    run_mask_decoder,
    run_prompt_encoder,
    tokenize,
)

# ── MemoryBank ────────────────────────────────────────────────────────────────


class MemoryBank:
    """Stores per-frame spatial features and object pointers for memory_attention.

    Conditioning frame (frame 0): kept forever, tpos = maskmem_tpos_enc[NUM_MASKMEM-1].
    Non-conditioning frames: rolling window of at most NON_COND_MAX recent frames,
    tpos based on relative age.

    memory_obj layout:
      [ cond_frame (HW) | spatial_0..k (HW each) | ptr_0..j (MULTIPLEX_COUNT each) ]
      T_mem = (1+k)*HW + j*MULTIPLEX_COUNT,  T_img = (1+k)*HW
    """

    NON_COND_MAX = NUM_MASKMEM - 1  # = 6 spatial non-cond frames
    NON_COND_PTR_MAX = (
        MAX_OBJ_PTRS - 1
    )  # = 15 non-cond ptr frames (1 slot reserved for cond)

    def __init__(self):
        self.cond_frame = None  # conditioning frame (frame 0), always kept
        self.spatial_frames = []  # non-conditioning sliding window, max NON_COND_MAX
        self.ptr_frames_nc = []  # non-cond ptr rolling window, max NON_COND_PTR_MAX

    def add(
        self, frame_idx, fpn2, pos2, mem_feat, mem_pos, all_ptrs, is_conditioning=False
    ):
        entry = dict(
            frame_idx=frame_idx,
            fpn2=fpn2,  # (1, 256, 72, 72)
            pos2=pos2,  # (1, 256, 72, 72)
            mem_feat=mem_feat,  # (1, 256, 72, 72)
            mem_pos=mem_pos,  # (1, 256, 72, 72)
            all_ptrs=all_ptrs.reshape(MULTIPLEX_COUNT, 256),  # (16, 256)
        )
        if is_conditioning:
            self.cond_frame = entry
        else:
            self.spatial_frames.append(entry)
            if len(self.spatial_frames) > self.NON_COND_MAX:
                self.spatial_frames.pop(0)
            # Cond ptr is kept separately; only non-cond ptrs roll off.
            self.ptr_frames_nc.append(entry)
            if len(self.ptr_frames_nc) > self.NON_COND_PTR_MAX:
                self.ptr_frames_nc.pop(0)


# ── Mask preprocessing ────────────────────────────────────────────────────────


def apply_interactive_mask_downsample(binary_mask_hw):
    """Conv2d(1,1,kernel=4,stride=4) at original resolution → (H//4, W//4)."""
    load_embeds()
    kernel = imd_weight_cache[0, 0]
    bias = float(imd_bias_cache[0])
    H, W = binary_mask_hw.shape
    pH = ((H + 3) // 4) * 4
    pW = ((W + 3) // 4) * 4
    if pH != H or pW != W:
        padded = np.zeros((pH, pW), dtype=np.float32)
        padded[:H, :W] = binary_mask_hw
    else:
        padded = binary_mask_hw
    out_H, out_W = pH // 4, pW // 4
    patches = padded.reshape(out_H, 4, out_W, 4)
    out = np.tensordot(patches, kernel, axes=([1, 3], [0, 1])) + bias
    return out.astype(np.float32)


def mask_for_prompt_encoder(binary_mask_hw, mask_input_size=(288, 288)):
    """Conv2d downsample + bilinear resize → (1, 1, H, W)."""
    ds = apply_interactive_mask_downsample(binary_mask_hw)
    resized = cv2.resize(
        ds, (mask_input_size[1], mask_input_size[0]), interpolation=cv2.INTER_LINEAR
    )
    return resized[np.newaxis, np.newaxis].astype(np.float32)


# ── 32-channel mask builders ──────────────────────────────────────────────────


def build_combined_32ch_mask(binary_masks_orig, is_conditioning):
    """
    (1, 32, MEMORY_MASK_SIZE, MEMORY_MASK_SIZE) for N objects.

    Channel layout mandated by the memory encoder:
      ch 0..N-1      : object mask probabilities (sigmoid scale+bias applied)
      ch N..15       : 0  (unused object slots)
      ch 16..16+N-1  : 1 if conditioning frame, else 0
      ch 16+N..31    : 0
    """
    N = len(binary_masks_orig)
    masks_32ch = np.zeros(
        (1, MASK_CHANNELS, MEMORY_MASK_SIZE, MEMORY_MASK_SIZE), dtype=np.float32
    )
    for k, bm in enumerate(binary_masks_orig):
        bm = bm.astype(np.float32)
        # convert binary {0,1} to logit space so sigmoid recovers near 0/1
        logit = bm * 20.0 - 10.0
        prob = sigmoid(logit) * SIGMOID_SCALE_FOR_MEM_ENC + SIGMOID_BIAS_FOR_MEM_ENC
        resized = tv_resize(prob, (MEMORY_MASK_SIZE, MEMORY_MASK_SIZE))
        masks_32ch[0, k] = resized
        masks_32ch[0, 16 + k] = 1.0 if is_conditioning else 0.0
    return masks_32ch


# ── Memory input assembly ─────────────────────────────────────────────────────


def build_combined_memory_inputs(memory_banks, frame_idx, maskmem_tpos_enc, models):
    """
    Returns (memory_obj, memory_obj_pos, memory_img, memory_img_pos).

    All banks store identical spatial features and full 16-slot ptr history.
    Bank 0 is representative for all banks.
    """
    return memory_banks[0].build_memory_inputs(frame_idx, maskmem_tpos_enc, models)


def run_memory_encoder(models, fpn2, masks_32ch):
    """
    fpn2       : (B, 256, 72, 72)
    masks_32ch : (B, 32, 1152, 1152)
    Returns    : vision_features(B,256,72,72), vision_pos_enc(B,256,72,72)
    """
    menc = models["mem_enc"]
    feed = {
        "pix_feat": fpn2.astype(np.float32),
        "masks": masks_32ch.astype(np.float32),
    }
    if not args.onnx:
        out = menc.predict(list(feed.values()))
    else:
        out = menc.run(None, feed)
    vision_features, vision_pos_enc = out
    return vision_features, vision_pos_enc


class Sam3Tracker:

    def __init__(self, models, maskmem_tpos_enc, no_obj_params, threshold=0.5):
        self.models = models
        self.maskmem_tpos_enc = maskmem_tpos_enc
        self.no_obj_params = no_obj_params
        self.threshold = threshold
        self.memory_banks = []  # list[MemoryBank], one per tracked object
        self.caption = None

    # ── state management ───────────────────────────────────────────────────────

    def reset(self):
        """Clear all tracked objects."""
        self.memory_banks = []

    # ── frame 0 initialisation ─────────────────────────────────────────────────

    def add_prompt(self, frame, caption):
        """
        Text grounding on frame 0; detects objects and initialises memory banks.

        Returns (scores, boxes, bin_masks) for the detected objects.
        bank_idxs are implicitly 0..N-1.
        """
        self.caption = caption
        models = self.models
        no_obj_params = self.no_obj_params

        orig_h, orig_w = frame.shape[:2]
        enc_out = run_encoder(models, preprocess(frame))
        (
            fpn0,
            fpn1,
            fpn2,
            pos0,
            pos1,
            pos2,
            prop_fpn0,
            prop_fpn1,
            prop_fpn2,
            prop_pos2,
        ) = enc_out

        text_tokens = tokenize(caption)
        gnd_out = run_grounding(
            models,
            fpn0,
            fpn1,
            fpn2,
            pos2,
            text_tokens,
            np.zeros((0, 1, 4), dtype=np.float32),
            np.zeros((0, 1), dtype=np.int64),
            np.zeros((1, 0), dtype=bool),
        )
        pred_masks_gnd, pred_boxes_gnd, pred_logits_gnd, presence_gnd = gnd_out

        scores_tmp, _, bin_masks_tmp = postprocess(
            pred_masks_gnd,
            pred_boxes_gnd,
            pred_logits_gnd,
            presence_gnd,
            orig_h,
            orig_w,
            self.threshold,
        )
        if len(scores_tmp) == 0:
            return np.zeros(0), np.zeros((0, 4)), np.zeros((0, orig_h, orig_w), bool)

        N = len(scores_tmp)

        # get per-object obj_ptr from the interactive decoder (one call per object)
        obj_ptrs = []
        for i in range(N):
            mfp = mask_for_prompt_encoder(
                bin_masks_tmp[i].astype(np.float32), mask_input_size=(288, 288)
            )
            coords = np.zeros((1, 1, 2), dtype=np.float32)
            labels = np.array([[-1]], dtype=np.int32)
            mask_enable = np.array([1], dtype=np.int32)
            sparse_emb, dense_emb, dense_pe = run_prompt_encoder(
                models, coords, labels, mfp, mask_enable
            )
            masks_dec, iou_pred, sam_tokens_out, _ = run_mask_decoder(
                models, prop_fpn2, dense_pe, sparse_emb, dense_emb, prop_fpn0, prop_fpn1
            )
            best_slot = int(np.argmax(iou_pred[0]))
            obj_ptrs.append(
                run_interactive_obj_ptr_proj(models, sam_tokens_out[:, best_slot, :])
            )

        # encode all N objects in a single combined call; all banks share these features
        combined_masks = build_combined_32ch_mask(
            [bin_masks_tmp[i] for i in range(N)], is_conditioning=True
        )
        mem_feat_combined, mem_pos_combined = run_memory_encoder(
            models, prop_fpn2, combined_masks
        )
        # add no-obj embeddings for unused slots (slots N..15) so the decoder
        # sees zero signal rather than an uninitialised embedding
        if no_obj_params is not None:
            mem_feat_combined = mem_feat_combined + no_obj_params[0][N:].sum(
                axis=0
            ).reshape(1, 256, 1, 1)

        # build combined 16-slot ptr block for the conditioning frame
        init_all_ptrs = np.zeros((MULTIPLEX_COUNT, 256), dtype=np.float32)
        for i in range(N):
            init_all_ptrs[i] = obj_ptrs[i].reshape(256)
        if no_obj_params is not None:
            _, W, b = no_obj_params
            for k in range(N, MULTIPLEX_COUNT):
                init_all_ptrs[k] = (
                    np.zeros((1, 256), dtype=np.float32) @ W.T + b
                ).reshape(256)

        # create MemoryBanks
        self.memory_banks = []
        all_scores, all_boxes, all_masks = [], [], []
        for i in range(N):
            mb = MemoryBank()
            self.memory_banks.append(mb)
            mb.add(
                0,
                prop_fpn2,
                prop_pos2,
                mem_feat_combined,
                mem_pos_combined,
                init_all_ptrs,
                is_conditioning=True,
            )

            bm = bin_masks_tmp[i]
            yx = np.where(bm)
            if len(yx[0]) == 0:
                continue
            y1, y2 = int(yx[0].min()), int(yx[0].max())
            x1, x2 = int(yx[1].min()), int(yx[1].max())
            all_scores.append(scores_tmp[i])
            all_boxes.append([x1, y1, x2, y2])
            all_masks.append(bm)

        return (
            np.array(all_scores, dtype=np.float32),
            np.array(all_boxes, dtype=np.float32),
            np.array(all_masks, dtype=bool),
        )

    # ── propagation ───────────────────────────────────────────────────────────

    def propagate_in_video(self, frame_paths, start_frame=1):
        """
        Forward pass; yields (frame_idx, scores, boxes, masks, obj_ids) per frame.

        obj_ids are 1-indexed object IDs stable across remove_object calls.
        Checks len(self.memory_banks) each frame, so remove_object between yields
        takes effect on the next frame.
        """
        models = self.models
        maskmem_tpos_enc = self.maskmem_tpos_enc
        no_obj_params = self.no_obj_params

        for frame_idx in range(start_frame, len(frame_paths)):
            N = len(self.memory_banks)
            if N == 0:
                return

            frame = cv2.imread(frame_paths[frame_idx])
            if frame is None:
                break
            orig_h, orig_w = frame.shape[:2]

            enc_out = run_encoder(models, preprocess(frame))
            fpn0, fpn1, fpn2 = enc_out[0], enc_out[1], enc_out[2]
            pos2 = enc_out[5]
            prop_fpn0, prop_fpn1, prop_fpn2, prop_pos2 = (
                enc_out[6],
                enc_out[7],
                enc_out[8],
                enc_out[9],
            )

            # 1. memory_attention
            memory_obj, memory_obj_pos, memory_img, memory_img_pos = (
                build_combined_memory_inputs(
                    self.memory_banks, frame_idx, maskmem_tpos_enc, models
                )
            )
