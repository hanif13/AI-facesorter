import argparse
import json
import logging
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
ort.disable_telemetry_events()
from tqdm import tqdm
from insightface.model_zoo import get_model
from src.common.logging_utils import setup_logging

def embed(faces_path: Path, model_path: Path, output: Path):
    if not model_path.is_file(): raise FileNotFoundError(model_path)
    faces = pd.read_parquet(faces_path)
    model = get_model(str(model_path), providers=["CPUExecutionProvider"])
    model.prepare(ctx_id=-1)
    vectors = []
    for row in tqdm(faces.itertuples(), total=len(faces), desc="Baseline embeddings"):
        crop = cv2.imread(row.crop_path)
        if crop is None: raise ValueError(f"Unreadable crop: {row.crop_path}")
        vector = model.get_feat(crop).reshape(-1).astype(np.float32)
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm <= 0: raise ValueError("Invalid embedding")
        vectors.append(vector / norm)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "embeddings.npy", np.stack(vectors) if vectors else np.empty((0, 512), np.float32))
    faces[["face_id"]].to_csv(output / "embedding_faces.csv", index=False)
    (output / "embedding_config.json").write_text(json.dumps({"embedder": "pretrained_baseline",
        "model": str(model_path.resolve()), "faces": len(faces), "dimension": 512}, indent=2))
    logging.info("embedder=pretrained_baseline faces=%d", len(faces))

def main():
    p = argparse.ArgumentParser(description="Comparison baseline ONLY: pretrained ArcFace R50")
    p.add_argument("--faces", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("outputs/models/models/buffalo_l/w600k_r50.onnx"))
    p.add_argument("--output", type=Path, default=Path("outputs/embeddings/baseline"))
    a = p.parse_args(); setup_logging("baseline_pretrained"); embed(a.faces, a.model, a.output)

if __name__ == "__main__": main()
