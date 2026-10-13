"""Three scorers with graceful degradation; heuristic maps to a 2-8 scale."""
import os

MODELS_DIR = os.getenv("AEVUM_SCORER_MODELS_DIR", "./models/scorers")
LIGHT = os.getenv("AEVUM_SCORER_LIGHT") == "1"  # force heuristic (CI-friendly)


class AestheticScorer:
    def __init__(self):
        self._mode = "heuristic"
        if LIGHT:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            self._clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            self._proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            head = os.path.join(MODELS_DIR, "aes-B32.pt")
            if os.path.exists(head):
                self._head = torch.load(head, map_location="cpu")
                self._mode = "full"
        except Exception:
            pass

    def score(self, image) -> float:
        if image is None:
            return 5.0
        if self._mode == "full":
            inputs = self._proc(images=image, return_tensors="pt")
            return float(self._head(self._clip.get_image_features(**inputs)))
        import numpy as np
        arr = np.asarray(image.convert("RGB").resize((224, 224)))
        variance_score = min(arr.std() / 64.0, 1.0)
        richness = min(len(np.unique(arr.reshape(-1, 3), axis=0)) / 2e4, 1.0)
        return round(float(2.0 + 6.0 * (0.5 * variance_score + 0.5 * richness)), 2)


class AnomalyScorer:
    def __init__(self):
        self._model = None
        if LIGHT:
            return
        try:
            from ultralytics import YOLO
            p = os.path.join(MODELS_DIR, "hand_yolov8n.pt")
            if os.path.exists(p):
                self._model = YOLO(p)
        except Exception:
            pass

    def anomalies(self, image):
        if image is None or self._model is None:
            return []
        r = self._model(image)[0]
        hands = sum(1 for b in r.boxes if int(b.cls) == 0 and float(b.conf) > 0.5)
        return [f"hands={hands}"] if hands > 2 else []


class CLIPScorer:
    def __init__(self):
        self._ok = False
        if LIGHT:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._ok = True
        except Exception:
            pass

    def aligned(self, image, prompt):
        if image is None or not self._ok:
            return True, 0.5
        import torch
        inputs = self._proc(text=[prompt], images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = self._clip(**inputs)
        sim = float(torch.softmax(out.logits_per_image, dim=1)[0][0])
        return sim >= 0.20, round(sim, 3)
