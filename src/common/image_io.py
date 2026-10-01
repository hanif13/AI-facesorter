from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import pillow_heif
pillow_heif.register_heif_opener()
EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff"}

def image_paths(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)

def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")

def load_bgr(path: Path) -> np.ndarray:
    return np.asarray(load_rgb(path))[:, :, ::-1].copy()
