"""Quality pipeline: anomaly + artifact + alignment + aesthetic -> Verdict."""
from dataclasses import dataclass, field

from aevum_scorer.artifact import ArtifactScorer
from aevum_scorer.scorers import AestheticScorer, AnomalyScorer, CLIPScorer


@dataclass
class Verdict:
    passed: bool
    score: float
    aesthetic: float = 0.0
    alignment: float = 0.0
    reasons: list = field(default_factory=list)


class QualityPipeline:
    def __init__(self):
        self.aes = AestheticScorer()
        self.anom = AnomalyScorer()
        self.clip = CLIPScorer()

    def check(self, image, prompt: str = "") -> Verdict:
        if image is None:
            return Verdict(True, 0.5, reasons=["image_unavailable"])
        reasons = self.anom.anomalies(image)
        reasons.extend(ArtifactScorer().check(image).reasons)
        aligned, sim = self.clip.aligned(image, prompt)
        if not aligned:
            reasons.append(f"clip_sim={sim}")
        aes_score = self.aes.score(image)
        return Verdict(passed=aligned and not reasons,
                       score=round(0.6 * sim + 0.4 * min(aes_score / 10.0, 1.0), 3),
                       aesthetic=round(float(aes_score), 3), alignment=sim, reasons=reasons)
