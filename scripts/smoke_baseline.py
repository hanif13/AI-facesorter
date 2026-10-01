"""Phase 1 baseline runner; this is not the final Phase 7 own-model pipeline."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Run Phase 1 baseline on a photo sample")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--run-name", default="smoke100")
    a = p.parse_args()
    if Path(a.run_name).name != a.run_name: p.error("Run name must be a single folder name")
    work = Path("outputs/embeddings") / a.run_name
    clusters = Path("outputs/clusters") / a.run_name
    sorted_dir = Path("outputs/sorted_photos") / a.run_name
    if any(path.exists() for path in [work, clusters, sorted_dir]):
        p.error("Run name already exists; use a new name")
    started = time.monotonic()
    commands = [
        ["src.detect_align", "--input", str(a.input), "--output", str(work), "--limit", str(a.limit)],
        ["src.baseline_pretrained", "--faces", str(work / "faces.parquet"), "--output", str(work)],
        ["src.cluster", "--embeddings", str(work / "embeddings.npy"), "--face-ids", str(work / "embedding_faces.csv"), "--output", str(clusters)],
        ["src.organize", "--faces", str(work / "faces.parquet"), "--assignments", str(clusters / "assignments.csv"),
         "--images", str(work / "images.csv"), "--source", str(a.input), "--output", str(sorted_dir)],
    ]
    for command in commands: subprocess.run([sys.executable, "-m", *command], check=True)
    summary = json.loads((sorted_dir / "summary.json").read_text())
    summary.update(runtime_seconds=time.monotonic() - started, embedder="pretrained_baseline",
                   detector="pretrained SCRFD", sample_limit=a.limit)
    (work / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
