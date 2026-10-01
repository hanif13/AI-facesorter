import argparse
from importlib.metadata import version
import json
import platform
import shutil
import sys
import subprocess
from pathlib import Path
from src.common.device import get_device
from src.common.logging_utils import setup_logging
import torch

def main():
    p = argparse.ArgumentParser(description="Verify required imports and actual MPS execution")
    p.add_argument("--output", type=Path, default=Path("outputs/logs/environment.json"))
    p.add_argument("--ram-gb", type=int, default=16, help="RAM reported by the human")
    a = p.parse_args(); setup_logging("environment")
    versions = {}
    # Keep ONNX/OpenCV import checks separate from MPS: combined teardown aborted on this host.
    subprocess.run([sys.executable, "-c", "import src; import torch, torchvision, numpy, pandas, sklearn, cv2, insightface, onnxruntime, pyarrow; print('All imports OK')"], check=True)
    for name in ["torch", "torchvision", "numpy", "pandas", "scikit-learn", "opencv-python", "insightface", "onnxruntime", "pyarrow"]:
        versions[name] = version(name)
    device = get_device()
    x = torch.randn(1024, 1024, device=device)
    value = float((x @ x).sum())
    result = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
              "ram_gb_human_reported": a.ram_gb, "free_disk_bytes": shutil.disk_usage(".").free,
              "versions": versions, "mps_built": torch.backends.mps.is_built(),
              "mps_available": torch.backends.mps.is_available(), "device": str(device),
              "matrix_operation_sum": value}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
