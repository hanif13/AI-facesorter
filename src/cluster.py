import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, DBSCAN, AgglomerativeClustering
from src.common.logging_utils import setup_logging

def cluster_vectors(x, algorithm="hdbscan", min_cluster_size=2, min_samples=1,
                    eps=0.45, distance_threshold=0.45, rescue_sim=0.45):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all(): raise ValueError("Invalid embeddings")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms <= 0).any(): raise ValueError("Zero embeddings")
    x = x / norms
    rescued = np.zeros(len(x), dtype=bool)
    if len(x) < max(2, min_cluster_size if algorithm == "hdbscan" else 2):
        return np.full(len(x), -1, dtype=int), rescued
    if algorithm == "hdbscan":
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                        metric="euclidean", cluster_selection_method="leaf")
    elif algorithm == "dbscan": model = DBSCAN(eps=eps, min_samples=2, metric="cosine")
    elif algorithm == "agglomerative":
        model = AgglomerativeClustering(linkage="average", metric="cosine",
                                       distance_threshold=distance_threshold, n_clusters=None)
    else: raise ValueError(algorithm)
    labels = model.fit_predict(x)
    ids = sorted(set(labels) - {-1})
    if ids and rescue_sim is not None:
        centers = np.stack([x[labels == i].mean(axis=0) for i in ids])
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
        for index in np.flatnonzero(labels == -1):
            sims = centers @ x[index]
            best = int(sims.argmax())
            if sims[best] >= rescue_sim:
                labels[index], rescued[index] = ids[best], True
    return labels, rescued

def main():
    p = argparse.ArgumentParser(description="Cluster normalized face embeddings")
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--face-ids", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/clusters/baseline"))
    p.add_argument("--algo", choices=["hdbscan", "dbscan", "agglomerative"], default="hdbscan")
    p.add_argument("--min-cluster-size", type=int, default=2)
    p.add_argument("--min-samples", type=int, default=1)
    p.add_argument("--eps", type=float, default=0.45)
    p.add_argument("--distance-threshold", type=float, default=0.45)
    p.add_argument("--rescue-sim", type=float, default=0.45)
    p.add_argument("--no-rescue", action="store_true")
    a = p.parse_args(); setup_logging("cluster")
    ids = pd.read_csv(a.face_ids)
    x = np.load(a.embeddings, allow_pickle=False)
    if len(ids) != len(x) or ids.face_id.duplicated().any(): raise ValueError("Face ID mismatch")
    labels, rescued = cluster_vectors(x, a.algo, a.min_cluster_size, a.min_samples, a.eps,
                                      a.distance_threshold, None if a.no_rescue else a.rescue_sim)
    a.output.mkdir(parents=True, exist_ok=True)
    ids.assign(cluster_id=labels, rescued=rescued).to_csv(a.output / "assignments.csv", index=False)
    (a.output / "config.json").write_text(json.dumps(vars(a), default=str, indent=2))
    logging.info("Faces=%d clusters=%d unknown=%d rescued=%d", len(x), len(set(labels)-{-1}),
                 int((labels == -1).sum()), int(rescued.sum()))

if __name__ == "__main__": main()
