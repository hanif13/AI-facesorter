import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import random
import numpy as np
import torch

def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description="Verify MPS and perform a matrix multiplication").parse_args()
    device = get_device()
    print({"device": str(device), "mps_available": torch.backends.mps.is_available(),
           "mps_built": torch.backends.mps.is_built()})
    x = torch.randn(1024, 1024, device=device)
    print(float((x @ x).sum()))
