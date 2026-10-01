import argparse
import hashlib
import json
import logging
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
ort.disable_telemetry_events()
from tqdm import tqdm
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
from src.common.image_io import image_paths, load_bgr
from src.common.logging_utils import setup_logging

COLUMNS = ["face_id", "image_path", "crop_path", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
           "det_score", "blur", "face_w", "face_h", "kps"]

def detect(input_dir: Path, output: Path, model_root: Path, det_score=0.5, min_face=40., min_blur=0., limit=None):
    output.mkdir(parents=True, exist_ok=True)
    crops = output / "crops"
    crops.mkdir(exist_ok=True)
    app = FaceAnalysis(name="buffalo_l", root=str(model_root), allowed_modules=["detection"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=det_score)
    rows, images = [], []
    paths = image_paths(input_dir)
    if limit is not None: paths = paths[:limit]
    for path in tqdm(paths, desc="Detect faces"):
        record = {"image_path": str(path.resolve()), "status": "ok", "detected": 0, "accepted": 0}
        try:
            original = load_bgr(path)
            scale = min(1., 2000 / max(original.shape[:2]))
            image = cv2.resize(original, None, fx=scale, fy=scale) if scale < 1 else original
            faces = app.get(image)
            record["detected"] = len(faces)
            for index, face in enumerate(faces):
                bbox = np.asarray(face.bbox) / scale
                kps = np.asarray(face.kps) / scale
                width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                crop = norm_crop(original, landmark=kps, image_size=112)
                blur = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                if face.det_score < det_score or min(width, height) < min_face or blur < min_blur:
                    continue
                face_id = hashlib.sha256(f"{path.resolve()}:{index}".encode()).hexdigest()[:20]
                crop_path = crops / f"{face_id}.jpg"
                if not cv2.imwrite(str(crop_path), crop):
                    raise OSError(f"Cannot write {crop_path}")
                rows.append([face_id, str(path.resolve()), str(crop_path.resolve()), *bbox.tolist(),
                             float(face.det_score), blur, float(width), float(height), json.dumps(kps.tolist())])
                record["accepted"] += 1
            if not faces:
                logging.info("Zero detected faces: %s", path)
        except (OSError, ValueError, cv2.error) as error:
            record.update(status="unreadable", error=str(error))
            logging.warning("Skipping %s: %s", path, error)
        images.append(record)
    pd.DataFrame(rows, columns=COLUMNS).to_parquet(output / "faces.parquet", index=False)
    pd.DataFrame(images).to_csv(output / "images.csv", index=False)
    (output / "detection_config.json").write_text(json.dumps({"input": str(input_dir.resolve()),
        "detector": "pretrained SCRFD buffalo_l", "det_score": det_score, "min_face": min_face,
        "min_blur": min_blur, "limit": limit, "images": len(images), "accepted_faces": len(rows)}, indent=2))
    logging.info("Images=%d accepted faces=%d", len(images), len(rows))

def main():
    p = argparse.ArgumentParser(description="Pretrained SCRFD detection and 5-point alignment")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/embeddings/baseline"))
    p.add_argument("--model-root", type=Path, default=Path("outputs/models"))
    p.add_argument("--det-score", type=float, default=0.5)
    p.add_argument("--min-face", type=float, default=40)
    p.add_argument("--min-blur", type=float, default=0)
    p.add_argument("--limit", type=int, help="Use the first N photos for a reproducible smoke test")
    a = p.parse_args()
    if not a.input.is_dir(): p.error("Input directory does not exist")
    if a.limit is not None and a.limit < 1: p.error("Limit must be positive")
    setup_logging("detect_align")
    detect(a.input, a.output, a.model_root, a.det_score, a.min_face, a.min_blur, a.limit)

if __name__ == "__main__": main()
