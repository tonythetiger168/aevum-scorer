"""Artifact detection: VAE tile seams, oversaturation, watermark-like corners."""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ArtifactReport:
    tile_seam: bool
    watermark: bool
    oversaturated: bool
    reasons: list = field(default_factory=list)


class ArtifactScorer:
    def check(self, image) -> ArtifactReport:
        if image is None:
            return ArtifactReport(False, False, False, [])
        arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
        reasons = []
        gray = arr.mean(axis=2)
        col_var = gray.var(axis=0)
        fft = np.abs(np.fft.rfft(col_var - col_var.mean()))
        peaks = sorted(fft[8:200], reverse=True)[:3]
        tile_seam = len(peaks) >= 2 and peaks[0] > 3 * np.median(fft[8:200])
        if tile_seam:
            reasons.append("periodic_64_128px (VAE tile seam)")
        sat = arr.max(axis=2) - arr.min(axis=2)
        oversaturated = float((sat > 0.95).mean()) > 0.15
        if oversaturated:
            reasons.append(f"oversaturation={float((sat > 0.95).mean()):.2f}")
        corner = gray[-64:, -64:]
        watermark = corner.std() < 0.02 and gray[:-64].std() > 0.15
        if watermark:
            reasons.append("regular_texture_bottom_right (possible watermark)")
        return ArtifactReport(tile_seam, watermark, oversaturated, reasons)
