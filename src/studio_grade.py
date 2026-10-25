"""Selective room-wall desaturation with a tracked skin-protection mask."""

import numpy as np


class StudioCommentaryGrade:
    def __init__(self, width, height):
        self.width, self.height = width, height
        y, x = np.ogrid[-1 : 1 : height * 1j, -1 : 1 : width * 1j]
        self.vignette = (1 - 0.25 * np.clip((x * x + y * y) / 1.5, 0, 1)).astype(np.float32)[:, :, None]
        self.vignette_rgb = np.repeat(self.vignette, 3, axis=2)
        # Gentle monotonic S curve, black/white endpoints retained.
        t = np.arange(256, dtype=np.float32) / 255
        self.curve = np.round(np.clip(t + 0.14 * (t - 0.5) * t * (1 - t) * 4, 0, 1) * 255).astype(np.uint8)
        self.last_mask = None
        self.last_anchor = None
        self.hue_weights = np.maximum(
            np.clip((np.arange(256, dtype=np.float32) - 125) / 20, 0, 1),
            np.clip((6 - np.arange(256, dtype=np.float32)) / 6, 0, 1),
        )

    def apply(self, frame, anchor):
        import cv2

        h, w = frame.shape[:2]
        if anchor and (
            self.last_anchor is None or max(abs(anchor[i] - self.last_anchor[i]) for i in (0, 1)) > 0.001
        ):
            # Ellipse includes forehead, ears and neck; no skin hue is used as a wall target.
            # Feather at quarter resolution, then resample the smooth mask. This
            # preserves the source-pixel feather width without a 115px full-HD blur.
            mw, mh = max(1, w // 4), max(1, h // 4)
            mask = np.zeros((mh, mw), np.uint8)
            cv2.ellipse(
                mask,
                (round(anchor[0] * mw), round((anchor[1] + 0.12) * mh)),
                (round(mw * 0.14), round(mh * 0.31)),
                0,
                0,
                360,
                255,
                -1,
            )
            mask = cv2.GaussianBlur(mask, (0, 0), max(0.75, mw / 100))
            self.last_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
            self.last_anchor = anchor
        face = self.last_mask if self.last_mask is not None else np.ones((h, w), np.float32)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        # Smooth pink/magenta range including red wrap, avoid orange skin range (hue 8–25).
        weight = cv2.LUT(hue, self.hue_weights) * (1 - face)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1].astype(np.float32), 1 - 0.35 * weight, dtype=cv2.CV_8U)
        graded = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        graded = cv2.LUT(graded, self.curve)
        # 5x5 low-pass unsharp, luma only; chroma remains untouched.
        ycc = cv2.cvtColor(graded, cv2.COLOR_BGR2YCrCb)
        sharp = cv2.addWeighted(ycc[:, :, 0], 1.6, cv2.GaussianBlur(ycc[:, :, 0], (5, 5), 0), -0.6, 0)
        ycc[:, :, 0] = cv2.add(
            ycc[:, :, 0], cv2.multiply(sharp.astype(np.float32) - ycc[:, :, 0], face), dtype=cv2.CV_8U
        )
        graded = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
        return cv2.multiply(graded, self.vignette_rgb, dtype=cv2.CV_8U)


def apply_studio_commentary_grade(frame, anchor, context=None):
    context = context or StudioCommentaryGrade(frame.shape[1], frame.shape[0])
    return context.apply(frame, anchor)
