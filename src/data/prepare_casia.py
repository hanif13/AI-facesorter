import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
import logging
from pathlib import Path
import shutil
import time
from zipfile import ZipFile
import numpy as np
from PIL import Image
from tqdm import tqdm
from src.common.logging_utils import setup_logging

def prepare(source: Path, output: Path, size=112, validation_images=200, seed=42,
            max_identities=None, max_per_identity=None, resume=False):
    if size not in {96, 112}: raise ValueError("Image size must be 96 or 112")
    if not source.is_file() and not source.is_dir(): raise FileNotFoundError(source)
    archive = ZipFile(source) if source.is_file() else None
    try:
        names = ([i.filename for i in archive.infolist() if not i.is_dir() and i.filename.lower().endswith((".jpg", ".jpeg", ".png"))]
                 if archive else [str(p.relative_to(source)) for p in source.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file()])
        if not names: raise ValueError("No identity-folder images found")
        grouped = {}
        for name in names:
            identity = Path(name).parent.name
            if not identity or identity == ".": raise ValueError("Identity folder missing")
            grouped.setdefault(identity, []).append(name)
        selected = sorted(grouped, key=lambda identity: (-len(grouped[identity]), identity))
        if max_identities is not None: selected = selected[:max_identities]
        identities = sorted(selected)
        names, labels = [], []
        for label, identity in enumerate(identities):
            items = sorted(grouped[identity])
            if max_per_identity is not None: items = items[:max_per_identity]
            names.extend(items); labels.extend([label] * len(items))
        labels = np.asarray(labels, dtype=np.int32)
        config = {"source": str(source.resolve()), "source_bytes": source.stat().st_size if archive else None,
                  "source_mtime_ns": source.stat().st_mtime_ns, "image_size": size, "images": len(names),
                  "classes": len(identities), "seed": seed, "validation_images": validation_images,
                  "max_identities": max_identities, "max_per_identity": max_per_identity,
                  "selection_rule": "identities ranked by image count; within each identity lexical path order",
                  "normalization": "RGB (x - 127.5) / 128"}
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        output.mkdir(parents=True, exist_ok=True)
        final = output / f"images_{size}.npy"
        partial = output / f"images_{size}.partial.npy"
        progress_path = output / "prep_progress.json"
        start = 0
        if final.exists(): raise FileExistsError("Prepared dataset already exists; choose a new output")
        if partial.exists():
            if not resume: raise FileExistsError("Partial dataset exists; use --resume")
            progress = json.loads(progress_path.read_text())
            if progress["fingerprint"] != fingerprint: raise ValueError("Resume configuration/source mismatch")
            start = progress["completed"]
        needed = len(names) * size * size * 3 + len(names) * 20 + 512 * 1024**2
        free = shutil.disk_usage(output).free
        if not partial.exists() and free < needed:
            raise OSError(f"Need {needed} bytes free, have {free}; use --max-identities/--max-per-identity")
        logging.info("images=%d classes=%d estimated_bytes=%d free_bytes=%d resume_at=%d",len(names),len(identities),needed,free,start)
        config["fingerprint"] = fingerprint
        (output / "prep_config.json").write_text(json.dumps(config, indent=2))
        mmap = np.lib.format.open_memmap(partial, mode="r+" if start or partial.exists() else "w+",
                                        dtype=np.uint8, shape=(len(names), size, size, 3))
        if not progress_path.exists(): progress_path.write_text(json.dumps({"fingerprint": fingerprint, "completed": 0}))
        started = time.monotonic()
        for index in tqdm(range(start, len(names)), initial=start, total=len(names), desc="Prepare CASIA"):
            try:
                raw = BytesIO(archive.read(names[index])) if archive else source / names[index]
                with Image.open(raw) as image:
                    image = image.convert("RGB")
                    if image.size != (size, size): image = image.resize((size, size), Image.Resampling.BILINEAR)
                    mmap[index] = np.asarray(image, dtype=np.uint8)
            except Exception:
                mmap.flush()
                progress_path.write_text(json.dumps({"fingerprint": fingerprint, "completed": index}))
                logging.exception("Cannot decode image index=%d path=%s; partial dataset retained", index,names[index])
                raise
            if (index + 1) % 10000 == 0:
                mmap.flush()
                progress_path.write_text(json.dumps({"fingerprint": fingerprint, "completed": index+1}))
        mmap.flush(); del mmap
        np.save(output / "labels.npy", labels, allow_pickle=False)
        rng = np.random.default_rng(seed)
        # At most one held-out image per identity; retain every class in training.
        candidates = [rng.integers(offset, offset+count) for offset,count in
                      zip(np.r_[0,np.cumsum(np.bincount(labels))[:-1]],np.bincount(labels)) if count >= 2]
        val = np.sort(rng.choice(candidates, size=min(validation_images,len(candidates)), replace=False))
        train = np.setdiff1d(np.arange(len(labels)), val)
        np.save(output / "train_indices.npy", train); np.save(output / "val_indices.npy", val)
        (output / "identities.json").write_text(json.dumps(identities))
        with (output / "image_index.tsv").open("w") as manifest:
            manifest.write("index\tlabel\tarchive_path\n")
            for index,(name,label) in enumerate(zip(names,labels)): manifest.write(f"{index}\t{label}\t{name}\n")
        elapsed = time.monotonic()-started
        stats = {**config, "train_images":len(train),"validation_images_actual":len(val),
                 "min_images_per_identity":int(np.bincount(labels).min()),
                 "max_images_per_identity":int(np.bincount(labels).max()),
                 "prep_seconds_this_run":elapsed,"image_bytes":partial.stat().st_size,
                 "source_contains_lfw":False,"source_crc_checked_during_decode":bool(archive)}
        assets = Path("outputs/report_assets"); assets.mkdir(parents=True,exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4)); ax.hist(np.bincount(labels), bins=60)
        ax.set(xlabel="Images per identity",ylabel="Number of identities",title="Training identity distribution")
        fig.tight_layout(); fig.savefig(assets/"dataset_stats.png",dpi=160); plt.close(fig)
        (output/"dataset_stats.json").write_text(json.dumps(stats,indent=2))
        partial.rename(final)
        progress_path.write_text(json.dumps({"fingerprint":fingerprint,"completed":len(names),"status":"complete"}))
        logging.info("Preparation complete: %s",json.dumps(stats))
        return stats
    finally:
        if archive: archive.close()

def main():
    p=argparse.ArgumentParser(description="Identity-folder ZIP/directory to uint8 RGB memmap; no MXNet needed")
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("data/casia_npy"))
    p.add_argument("--image-size",type=int,choices=[96,112],default=112)
    p.add_argument("--validation-images",type=int,default=200)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--max-identities",type=int)
    p.add_argument("--max-per-identity",type=int)
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.validation_images<0: p.error("validation-images must be nonnegative")
    if any(x is not None and x<2 for x in [a.max_identities,a.max_per_identity]): p.error("Subset limits must be at least 2")
    setup_logging("prepare_casia")
    prepare(a.source,a.output,a.image_size,a.validation_images,a.seed,a.max_identities,a.max_per_identity,a.resume)

if __name__=="__main__": main()
