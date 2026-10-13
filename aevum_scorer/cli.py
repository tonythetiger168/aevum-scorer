import argparse
import json
import sys

from PIL import Image

from aevum_scorer.pipeline import QualityPipeline


def main():
    p = argparse.ArgumentParser(prog="aevum-scorer",
                                description="Quality scoring for AI-generated images")
    p.add_argument("image", help="path to image file")
    p.add_argument("--prompt", default="", help="prompt used to generate the image")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args()

    img = Image.open(args.image)
    v = QualityPipeline().check(img, args.prompt)
    out = {"passed": v.passed, "score": v.score, "aesthetic": v.aesthetic,
           "alignment": v.alignment, "reasons": v.reasons}
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(f"{'PASS' if v.passed else 'FAIL'} score={v.score} "
              f"aesthetic={v.aesthetic} alignment={v.alignment}")
        for r in v.reasons:
            print(f"  - {r}")
    sys.exit(0 if v.passed else 1)


if __name__ == "__main__":
    main()
