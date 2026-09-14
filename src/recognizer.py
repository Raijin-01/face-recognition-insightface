from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:
    """Face recognition using InsightFace detection + ArcFace embeddings."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640),
        providers: Optional[list[str]] = None,
        threshold: float = 0.45,
    ) -> None:
        self.threshold = threshold
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=det_size)
        self.known_embeddings: Dict[str, np.ndarray] = {}

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Cannot normalize a zero embedding.")
        return embedding / norm

    def embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        faces = self.app.get(image)
        if not faces:
            return None
        # Use the largest detected face as the reference face.
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return self._normalize(face.embedding)

    def load_known_faces(self, root: str | Path) -> Dict[str, np.ndarray]:
        """Build one normalized reference embedding per identity.

        Directory layout: root/person_name/*.jpg|*.jpeg|*.png
        Multiple images for one person are averaged before normalization.
        """
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Known-face directory not found: {root}")

        result: Dict[str, np.ndarray] = {}
        for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            embeddings = []
            for path in sorted(person_dir.iterdir()):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                image = cv2.imread(str(path))
                if image is None:
                    continue
                emb = self.embedding(image)
                if emb is not None:
                    embeddings.append(emb)

            if embeddings:
                result[person_dir.name] = self._normalize(np.mean(embeddings, axis=0))

        self.known_embeddings = result
        return result

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = FaceRecognizer._normalize(a)
        b = FaceRecognizer._normalize(b)
        return float(np.dot(a, b))

    def recognize(self, embedding: np.ndarray) -> tuple[str, float]:
        if not self.known_embeddings:
            return "Unknown", 0.0

        scores = {
            name: self.cosine_similarity(embedding, reference)
            for name, reference in self.known_embeddings.items()
        }
        name, score = max(scores.items(), key=lambda item: item[1])
        return (name, score) if score >= self.threshold else ("Unknown", score)

    def recognize_image(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], str, float]]:
        """Detect and recognize every face in an image."""
        results = []
        for face in self.app.get(image):
            name, score = self.recognize(self._normalize(face.embedding))
            x1, y1, x2, y2 = map(int, face.bbox)
            results.append(((x1, y1, x2, y2), name, score))
        return results
