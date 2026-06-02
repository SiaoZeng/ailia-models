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



class Sam3Tracker:

    def __init__(self, models, maskmem_tpos_enc, no_obj_params, threshold=0.5):
        self.models = models
        self.maskmem_tpos_enc = maskmem_tpos_enc
        self.no_obj_params = no_obj_params
        self.threshold = threshold
        self.memory_banks = []  # list[MemoryBank], one per tracked object

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

