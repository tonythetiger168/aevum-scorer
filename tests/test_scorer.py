import os
import sys

sys.path.insert(0, ".")
os.environ["AEVUM_SCORER_LIGHT"] = "1"  # deterministic heuristic mode

import numpy as np
from PIL import Image

from aevum_scorer import QualityPipeline, AestheticScorer, ArtifactScorer


def noisy_img(seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray((rng.random((256, 256, 3)) * 255).astype("uint8"))


def flat_img():
    return Image.new("RGB", (256, 256), (128, 128, 128))


def saturated_img():
    rng = np.random.default_rng(1)
    arr = np.zeros((256, 256, 3), dtype="uint8")
    mask = rng.random((256, 256)) > 0.5
    arr[mask] = [255, 0, 0]
    arr[~mask] = [0, 0, 255]
    return Image.fromarray(arr)


def test_heuristic_range():
    s = AestheticScorer().score(noisy_img())
    assert 2.0 <= s <= 8.0, s
    print(f"PASS heuristic range: {s}")


def test_noisy_beats_flat():
    a = AestheticScorer()
    assert a.score(noisy_img()) > a.score(flat_img())
    print("PASS noisy > flat")


def test_artifact_oversaturation():
    r = ArtifactScorer().check(saturated_img())
    assert r.oversaturated, r
    assert any("oversaturation" in x for x in r.reasons)
    print("PASS oversaturation detected")


def test_artifact_clean_flat():
    r = ArtifactScorer().check(flat_img())
    assert not r.oversaturated and not r.watermark and not r.tile_seam
    print("PASS flat image clean")


def test_pipeline_verdict_fields():
    v = QualityPipeline().check(flat_img(), "a gray square")
    assert isinstance(v.passed, bool)
    assert 0.0 <= v.score <= 1.0
    assert v.reasons == []
    print(f"PASS pipeline verdict: {v}")


def test_pipeline_none_image():
    v = QualityPipeline().check(None, "")
    assert v.passed and "image_unavailable" in v.reasons
    print("PASS none image")


def test_cli(tmp_path=None):
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.png")
        noisy_img().save(p)
        r = subprocess.run([sys.executable, "-m", "aevum_scorer.cli", p, "--json"],
                           capture_output=True, text=True, cwd=".")
        assert r.returncode == 0, r.stderr
        import json
        out = json.loads(r.stdout)
        assert out["passed"] is True and out["score"] > 0
    print("PASS cli")


if __name__ == "__main__":
    test_heuristic_range(); test_noisy_beats_flat()
    test_artifact_oversaturation(); test_artifact_clean_flat()
    test_pipeline_verdict_fields(); test_pipeline_none_image(); test_cli()
    print("ALL SCORER TESTS PASSED")
