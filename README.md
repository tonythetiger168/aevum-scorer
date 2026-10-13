# aevum-scorer

Quality scoring for AI-generated images — aesthetic, anatomy anomalies
(>2 hands), prompt alignment, and artifact detection (VAE tile seams,
oversaturation, watermark-like textures) in one pipeline.

    pip install aevum-scorer
    aevum-scorer output.png --prompt "a cat on a bicycle" --json

No GPU required. With zero extra dependencies it runs heuristic scorers,
so it works in CI, serverless, anywhere. Install the `full` extra for
CLIP alignment, LAION-style aesthetic head, and YOLO hand detection:

    pip install aevum-scorer[full]
    # models: put aes-B32.pt and hand_yolov8n.pt in ./models/scorers/
    # or set AEVUM_SCORER_MODELS_DIR

## Python API

```python
from PIL import Image
from aevum_scorer import QualityPipeline

v = QualityPipeline().check(Image.open("output.png"), prompt="a cat")
v.passed      # bool
v.score       # 0..1 combined
v.aesthetic   # 2..8 heuristic, or model head output in full mode
v.alignment   # CLIP similarity in full mode, 0.5 neutral otherwise
v.reasons     # e.g. ["hands=3", "periodic_64_128px (VAE tile seam)"]
```

Set `AEVUM_SCORER_LIGHT=1` to force heuristic mode even when torch is
installed (deterministic CI).

## Honesty note
Heuristic fallback is honest about being heuristic. Full mode details
(CLIP + YOLO + aesthetic head) are in the source; benchmarks welcome as
PRs — methodology matters more than numbers.

## See also
- [aevum-cost](https://github.com/aevum-ai/aevum-cost) — real GPU cost per image/second
- [aevum-workflows](https://github.com/aevum-ai/aevum-workflows) — 336 ComfyUI workflows

---
aevum-scorer is one module of [Aevum](https://aevum.ai?utm_source=scorer),
the self-hosted video generation platform — the platform adds auto-retry,
model routing, and an audit trail on top of this pipeline. The scorer is
Apache-2.0 and free forever.
