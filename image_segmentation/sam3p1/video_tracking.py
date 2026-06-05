import cv2
import numpy as np
from sam3p1 import (
    postprocess,
    preprocess,
    run_encoder,
    run_grounding,
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


# ── Prompt / decoder / memory runners ────────────────────────────────────────


def run_mask_decoder(
    models, image_embeddings, image_pe, sparse_emb, dense_emb, fpn0, fpn1
):
    """
    Returns masks(B,4,288,288), iou_pred(B,4), sam_tokens_out(B,4,256), obj_score(B,1).
    high_res_features1 = fpn0 (288×288), high_res_features2 = fpn1 (144×144).
    """
    dec = models["mask_dec"]
    feed = {
        "image_embeddings": image_embeddings.astype(np.float32),
        "image_pe": image_pe.astype(np.float32),
        "sparse_prompt_embeddings": sparse_emb.astype(np.float32),
        "dense_prompt_embeddings": dense_emb.astype(np.float32),
        "high_res_features1": fpn0.astype(np.float32),
        "high_res_features2": fpn1.astype(np.float32),
    }
    if not args.onnx:
        out = dec.predict(list(feed.values()))
    else:
        out = dec.run(None, feed)
    masks, iou_pred, sam_tokens_out, object_score_logits = out
    return masks, iou_pred, sam_tokens_out, object_score_logits


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


def run_interactive_obj_ptr_proj(models, sam_tokens_first):
    """Interactive obj_ptr_proj for init frame: sam_tokens_first (B, 256) → (B, 256).

    Uses interactive_obj_ptr_proj (different weights from obj_ptr_proj)
    when is_interactive=True (i.e. the first frame with a prompt).
    """
    proj = models["iobj_proj"]
    x = sam_tokens_first.astype(np.float32)
    if not args.onnx:
        out = proj.predict([x])
    else:
        out = proj.run(None, {"x": x})
    return out[0]  # (B, 256)


# ── Sam3Tracker ───────────────────────────────────────────────────────────────


class Sam3Tracker:
    """
    SAM 3.1 ONNX tracker.

    Usage:
        tracker = Sam3Tracker(models, maskmem_tpos_enc, no_obj_params, threshold=0.5)

        # frame 0
        scores, boxes, masks = tracker.add_prompt(frame, caption)
        # or
        scores, boxes, masks = tracker.add_prompt_interactive(frame, box=box)

        # frames 1 → N
        for fi, scores, boxes, masks, obj_ids in tracker.propagate_in_video(frame_paths):
            ...

        # optional: drop an object
        tracker.remove_object(obj_idx)

        # start over
        tracker.reset()
    """

    def __init__(self, models, maskmem_tpos_enc, no_obj_params, threshold=0.5):
        self.models = models
        self.maskmem_tpos_enc = maskmem_tpos_enc
        self.no_obj_params = no_obj_params
        self.threshold = threshold
        self.memory_banks = []  # list[MemoryBank], one per tracked object
        self.caption = None
        # Per-object metadata (parallel to memory_banks)
        self.obj_ids = []  # stable 1-indexed IDs (never reused)
        self.next_id = 1
        self.keep_alive = []  # int: +1 matched / -1 unmatched, clamped [MIN,MAX]
        self.consecutive_det_count = (
            []
        )  # int: consecutive frames matched by a detection
        self.confirmed = []  # bool: True after MASKLET_CONFIRM_N consecutive matches
        self.add_frame = []  # int: frame index when first added
        self.unmatch_total = []  # int: total unmatched frames (never resets)
        self.pairwise_overlap = np.zeros(
            (0, 0), dtype=np.int32
        )  # (N, N) overlap counts
        self.current_frame = 0

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

        # create MemoryBanks and per-object state
        self.memory_banks = []
        self.obj_ids = []
        self.next_id = 1
        self.keep_alive = []
        self.consecutive_det_count = []
        self.confirmed = []
        self.add_frame = []
        self.unmatch_total = []
        self.pairwise_overlap = np.zeros((0, 0), dtype=np.int32)
        self.current_frame = 0

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
            # Frame-0 prompt objects: pre-confirmed (INIT_TRK_KEEP_ALIVE=0,
            # confirmed=True: frame-0 prompt objects skip the confirmation wait
            self._append_object_state(
                frame_idx=0, confirmed=True, keep_alive_init=INIT_TRK_KEEP_ALIVE
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

        Tracking parameters match Sam3MultiplexTrackingWithInteractivity in model_builder.py:
          assoc_iou_thresh=0.1 (IoM), new_det_thresh=0.65,
          masklet_confirmation_consecutive_det_thresh=3,
          hotstart_delay=15, hotstart_unmatch_thresh=8, hotstart_dup_thresh=8,
          suppress_unmatched_only_within_hotstart=False,
          det_nms_thresh=0.1 (IoM), score_threshold_detection=0.4
        """
        models = self.models
        maskmem_tpos_enc = self.maskmem_tpos_enc
        no_obj_params = self.no_obj_params

        # frame_idx → set of obj_ids still unconfirmed at that frame (for lookahead)
        unconfirmed_ids_per_frame = {}
        # (frame_idx, candidates) waiting until UNCONFIRMED_STATUS_DELAY future frames exist
        pending_outputs = []
        last_frame_idx = len(frame_paths) - 1

        for frame_idx in range(start_frame, len(frame_paths)):
            N = len(self.memory_banks)
            if N == 0:
                return

            frame = cv2.imread(frame_paths[frame_idx])
            if frame is None:
                break
            orig_h, orig_w = frame.shape[:2]

            # ── Step 1: image encoder ─────────────────────────────────────
            enc_out = run_encoder(models, preprocess(frame))
            fpn0, fpn1, fpn2 = enc_out[0], enc_out[1], enc_out[2]
            pos2 = enc_out[5]
            prop_fpn0, prop_fpn1, prop_fpn2, prop_pos2 = (
                enc_out[6],
                enc_out[7],
                enc_out[8],
                enc_out[9],
            )

            # ── Step 2: memory attention ──────────────────────────────────
            memory_obj, memory_obj_pos, memory_img, memory_img_pos = (
                build_combined_memory_inputs(
                    self.memory_banks, frame_idx, maskmem_tpos_enc, models
                )
            )
            curr_flat = prop_fpn2.reshape(1, 256, -1).transpose(2, 0, 1)
            curr_pos_flat = prop_pos2.reshape(1, 256, -1).transpose(2, 0, 1)
            pix_feat = run_memory_attention(
                models,
                curr_obj=curr_flat,
                curr_obj_pos=curr_pos_flat,
                curr_img=curr_flat,
                memory_obj=memory_obj,
                memory_obj_pos=memory_obj_pos,
                memory_img=memory_img,
                memory_img_pos=memory_img_pos,
            )

            # ── Step 3: tracking mask decoder ─────────────────────────────
            extra_embed = self.build_extra_embed(N)
            img_emb = pix_feat.transpose(1, 2, 0).reshape(1, 256, 72, 72)
            masks_all, iou_all, sam_tokens_all, obj_scores_all = (
                run_tracking_mask_decoder(
                    models, img_emb, prop_fpn0, prop_fpn1, extra_embed
                )
            )

            # ── Step 4: per-slot results + ptrs ───────────────────────────
            is_app_flags = []
            logit_bests = []  # (N,) of (288,288) tracking logit masks
            all_ptrs = np.zeros((MULTIPLEX_COUNT, 256), dtype=np.float32)

            for k in range(MULTIPLEX_COUNT):
                best_slot_k = int(np.argmax(iou_all[0, k]))
                is_app_k = k < N and float(obj_scores_all[0, k]) > OBJ_SCORE_THRESHOLD
                obj_ptr_k = run_obj_ptr_proj(
                    models,
                    sam_tokens_all[:, k, best_slot_k : best_slot_k + 1, :].reshape(
                        1, 256
                    ),
                )
                if not is_app_k and no_obj_params is not None:
                    _, W, b = no_obj_params
                    obj_ptr_k = (obj_ptr_k @ W.T + b).astype(np.float32)
                all_ptrs[k] = obj_ptr_k.reshape(256)
                if k < N:
                    logit_bests.append(masks_all[0, k, best_slot_k])
                    is_app_flags.append(is_app_k)

            # 非出現オブジェクトも実際のトラッキングマスクを使う。
            # ゼロマスクにすると新規検出との重複判定が正しく機能しない。
            existing_bin_masks = np.array(
                [lb > 0 for lb in logit_bests], dtype=bool
            )  # (N, 288, 288)

            # ── Step 5: grounding (NMS + association) ────────────────────
            new_cand_scores = []  # float, scores of accepted new detections
            new_cand_bin_masks = []  # (orig_h, orig_w) full-res binary, for output
            new_cand_raw_masks = []  # (288, 288) logit, for memory encoder
            new_obj_ptrs = []  # interactive obj_ptr per new object
            det_to_matched_trk = {}  # d → [k] for pairwise overlap tracking
            im_mask = np.zeros((0, N), dtype=bool)

            if self.caption is not None and N < MULTIPLEX_COUNT:
                text_tokens = tokenize(self.caption)
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

                # Filter at SCORE_THRESH_DET (= score_threshold_detection)
                out_probs = sigmoid(pred_logits_gnd[0, :, 0]) * sigmoid(
                    presence_gnd[0, 0]
                )
                gnd_keep = out_probs > SCORE_THRESH_DET
                gnd_keep_idx = np.where(gnd_keep)[0]

                if len(gnd_keep_idx) > 0:
                    raw_k = pred_masks_gnd[0][gnd_keep]  # (K, 288, 288) logit
                    scores_k = out_probs[gnd_keep]  # (K,)

                    # NMS with IoM (= det_nms_thresh=0.1, det_nms_use_iom=True)
                    nms_keep = nms_masks_iom(raw_k, scores_k, 0.0, DET_NMS_IOM_THRESH)
                    raw_nms = raw_k[nms_keep]  # (K_nms, 288, 288)
                    scores_nms = scores_k[nms_keep]  # (K_nms,)

                    # Association: IoM matrix (K_nms, N)
                    # Source: _associate_det_trk_compilable with use_iom_recondition=True
                    if N > 0:
                        iom_mat = mask_iom_matrix(
                            (raw_nms > 0), existing_bin_masks
                        )  # (K_nms, N)
                    else:
                        iom_mat = np.zeros((len(scores_nms), 0), np.float32)

                    im_mask = iom_mat >= ASSOC_IOM_THRESH  # (K_nms, N)

                    # det_to_matched_trk for pairwise overlap counting
                    for d in range(len(scores_nms)):
                        matched_ks = [k for k in range(N) if im_mask[d, k]]
                        if matched_ks:
                            det_to_matched_trk[d] = matched_ks

                    # is_new_det: score >= NEW_DET_THRESH AND IoM < ASSOC_IOM_THRESH with ALL tracks
                    is_new = (scores_nms >= NEW_DET_THRESH) & ~im_mask.any(axis=1)

                    for d in range(len(scores_nms)):
                        if not is_new[d]:
                            continue
                        if N + len(new_cand_bin_masks) >= MULTIPLEX_COUNT:
                            break
                        # full-res binary for output
                        bin_full = (
                            sigmoid(
                                cv2.resize(
                                    raw_nms[d],
                                    (orig_w, orig_h),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                            )
                            > 0.5
                        )
                        # get obj_ptr via interactive decoder
                        mfp = mask_for_prompt_encoder(
                            bin_full.astype(np.float32), mask_input_size=(288, 288)
                        )
                        coords = np.zeros((1, 1, 2), dtype=np.float32)
                        labels = np.array([[-1]], dtype=np.int32)
                        mask_enable = np.array([1], dtype=np.int32)
                        sparse_emb, dense_emb, dense_pe = run_prompt_encoder(
                            models, coords, labels, mfp, mask_enable
                        )
                        masks_dec, iou_pred, sam_tokens_out, _ = run_mask_decoder(
                            models,
                            prop_fpn2,
                            dense_pe,
                            sparse_emb,
                            dense_emb,
                            prop_fpn0,
                            prop_fpn1,
                        )
                        best = int(np.argmax(iou_pred[0]))
                        new_obj_ptrs.append(
                            run_interactive_obj_ptr_proj(
                                models, sam_tokens_out[:, best, :]
                            )
                        )
                        new_cand_scores.append(float(scores_nms[d]))
                        new_cand_bin_masks.append(bin_full)
                        new_cand_raw_masks.append(raw_nms[d])

            # ── Step 6: update per-object tracking state ──────────────────
            # Source: _process_hotstart (CPU version) + update_masklet_confirmation_status
            trk_is_matched = (
                im_mask.any(axis=0) if im_mask.shape[0] > 0 else np.zeros(N, dtype=bool)
            )
            trk_is_nonempty = (
                existing_bin_masks.any(axis=(1, 2))
                if N > 0
                else np.array([], dtype=bool)
            )
            trk_is_unmatched = trk_is_nonempty & ~trk_is_matched

            for k in range(N):
                if trk_is_matched[k]:
                    self.keep_alive[k] = min(TRK_KEEP_ALIVE_MAX, self.keep_alive[k] + 1)
                    self.consecutive_det_count[k] += 1
                else:
                    self.keep_alive[k] = max(TRK_KEEP_ALIVE_MIN, self.keep_alive[k] - 1)
                    self.consecutive_det_count[k] = 0
                if trk_is_unmatched[k]:
                    self.unmatch_total[k] += 1
                if self.consecutive_det_count[k] >= MASKLET_CONFIRM_N:
                    self.confirmed[k] = True

            # Update pairwise overlap counts for hotstart duplicate detection
            for d, matched_ks in det_to_matched_trk.items():
                if len(matched_ks) >= 2:
                    first_k = min(matched_ks, key=lambda k: self.add_frame[k])
                    for other_k in matched_ks:
                        if other_k != first_k:
                            self.pairwise_overlap[first_k, other_k] += 1

            # ── Step 7: hotstart removal ───────────────────────────────────
            # Source: _process_hotstart in sam3_video_base.py
            # suppress_unmatched_only_within_hotstart=False → keep_alive suppression always
            to_remove = []
            for k in range(N):
                frames_since_add = frame_idx - self.add_frame[k]
                is_within_hotstart = frames_since_add < HOTSTART_DELAY
                if not is_within_hotstart:
                    continue
                if self.unmatch_total[k] >= HOTSTART_UNMATCH_THRESH:
                    to_remove.append(k)
                    continue
                # Remove by pairwise overlap with any earlier object
                for j in range(N):
                    if self.add_frame[j] < self.add_frame[k]:
                        if self.pairwise_overlap[j, k] >= HOTSTART_DUP_THRESH:
                            to_remove.append(k)
                            break

            # suppress_by_keep_alive: keep_alive <= 0
            # (suppress_unmatched_only_within_hotstart=False → always active)
            suppress_set = {k for k in range(N) if self.keep_alive[k] <= 0}

            # ── Step 8: memory encoder (runs on ALL N + M_new objects before removal) ─
            M = len(new_cand_bin_masks)
            total_N = N + M
            combined_masks = np.zeros(
                (1, MASK_CHANNELS, MEMORY_MASK_SIZE, MEMORY_MASK_SIZE), dtype=np.float32
            )
            for k, logit in enumerate(logit_bests):
                # 非出現オブジェクトも実際のlogitをそのまま使う。
                # 出現フラグはno_obj_embedで別途伝達されるため、マスク側をゼロにする必要はない。
                prob = (
                    sigmoid(logit) * SIGMOID_SCALE_FOR_MEM_ENC
                    + SIGMOID_BIAS_FOR_MEM_ENC
                )
                combined_masks[0, k] = tv_resize(
                    prob, (MEMORY_MASK_SIZE, MEMORY_MASK_SIZE)
                )
                combined_masks[0, 16 + k] = 0.0
            for i, raw_m in enumerate(new_cand_raw_masks):
                logit = raw_m  # already logit space
                prob = (
                    sigmoid(logit) * SIGMOID_SCALE_FOR_MEM_ENC
                    + SIGMOID_BIAS_FOR_MEM_ENC
                )
                combined_masks[0, N + i] = tv_resize(
                    prob, (MEMORY_MASK_SIZE, MEMORY_MASK_SIZE)
                )
                combined_masks[0, 16 + N + i] = 1.0

            mem_feat_combined, mem_pos_combined = run_memory_encoder(
                models, prop_fpn2, combined_masks
            )
            if no_obj_params is not None:
                no_obj_embed_all = no_obj_params[0]
                mem_feat_combined = mem_feat_combined + no_obj_embed_all[total_N:].sum(
                    axis=0
                ).reshape(1, 256, 1, 1)
                for k in range(N):
                    if not is_app_flags[k]:
                        mem_feat_combined = mem_feat_combined + no_obj_embed_all[
                            k
                        ].reshape(1, 256, 1, 1)

            # ── Step 9: remove objects (compact state) ────────────────────
            to_remove_set = set(to_remove)
            remaining_old_slots = [k for k in range(N) if k not in to_remove_set]
            for k in sorted(to_remove_set, reverse=True):
                self.memory_banks.pop(k)
                self._remove_object_state(k)
            N_remaining = len(remaining_old_slots)

            # Build updated all_ptrs for remaining objects
            updated_all_ptrs = np.zeros((MULTIPLEX_COUNT, 256), dtype=np.float32)
            for new_k, old_k in enumerate(remaining_old_slots):
                updated_all_ptrs[new_k] = all_ptrs[old_k]
            if no_obj_params is not None:
                _, W, b = no_obj_params
                for k in range(N_remaining, MULTIPLEX_COUNT):
                    updated_all_ptrs[k] = (
                        np.zeros((1, 256), dtype=np.float32) @ W.T + b
                    ).reshape(256)

            # ── Step 10: update memory banks for remaining objects ─────────
            for mb in self.memory_banks:
                mb.add(
                    frame_idx,
                    prop_fpn2,
                    prop_pos2,
                    mem_feat_combined,
                    mem_pos_combined,
                    updated_all_ptrs,
                )

            # ── Step 11: add new objects ───────────────────────────────────
            for i, new_ptr in enumerate(new_obj_ptrs):
                slot = N_remaining + i
                cond_ptrs = updated_all_ptrs.copy()
                cond_ptrs[slot] = new_ptr.reshape(256)
                mb = MemoryBank()
                mb.add(
                    frame_idx,
                    prop_fpn2,
                    prop_pos2,
                    mem_feat_combined,
                    mem_pos_combined,
                    cond_ptrs,
                    is_conditioning=True,
                )
                self.memory_banks.append(mb)
                # init_trk_keep_alive=0: new objects start suppressed until first match
                self._append_object_state(
                    frame_idx=frame_idx,
                    confirmed=False,
                    keep_alive_init=INIT_TRK_KEEP_ALIVE,
                )

            # ── Step 12: collect output ────────────────────────────────────
            # Candidates: suppress + is_app pass. The confirmed check is deferred
            # by UNCONFIRMED_STATUS_DELAY frames (PyTorch unconfirmed_status_delay
            # lookahead): output frame N only if the object is confirmed at frame
            # N + UNCONFIRMED_STATUS_DELAY.
            candidates = []
            for new_k, old_k in enumerate(remaining_old_slots):
                if old_k in suppress_set:
                    continue
                if not is_app_flags[old_k]:
                    continue
                binary_mask = (
                    sigmoid(
                        tv_resize(logit_bests[old_k], (orig_h, orig_w), antialias=False)
                    )
                    > 0.5
                )
                yx = np.where(binary_mask)
                if len(yx[0]) == 0:
                    continue
                y1, y2 = int(yx[0].min()), int(yx[0].max())
                x1, x2 = int(yx[1].min()), int(yx[1].max())
                candidates.append(
                    (
                        self.obj_ids[new_k],
                        float(sigmoid(float(obj_scores_all[0, old_k]))),
                        [x1, y1, x2, y2],
                        binary_mask,
                    )
                )

            # Record which obj_ids are unconfirmed this frame (for future lookahead)
            unconfirmed_ids_per_frame[frame_idx] = {
                self.obj_ids[new_k]
                for new_k in range(len(remaining_old_slots))
                if not self.confirmed[new_k]
            }

            pending_outputs.append((frame_idx, candidates))
            self.current_frame = frame_idx

            # Yield oldest buffered frame when its lookahead frame has been processed,
            # or flush all remaining entries on the last frame.
            while len(pending_outputs) > UNCONFIRMED_STATUS_DELAY or (
                frame_idx == last_frame_idx and pending_outputs
            ):
                yield_frame_idx, yield_candidates = pending_outputs.pop(0)
                lookup_idx = min(yield_frame_idx + UNCONFIRMED_STATUS_DELAY, frame_idx)
                hidden_ids = unconfirmed_ids_per_frame.get(lookup_idx, set())
                all_scores_out, all_boxes_out, all_masks_out, obj_ids_out = (
                    [],
                    [],
                    [],
                    [],
                )
                for obj_id, score, box, mask in yield_candidates:
                    if obj_id in hidden_ids:
                        continue
                    all_scores_out.append(score)
                    all_boxes_out.append(box)
                    all_masks_out.append(mask)
                    obj_ids_out.append(obj_id)
                yield (
                    yield_frame_idx,
                    np.array(all_scores_out, dtype=np.float32),
                    np.array(all_boxes_out, dtype=np.float32),
                    np.array(all_masks_out, dtype=bool),
                    obj_ids_out,
                )
