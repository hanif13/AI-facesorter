import argparse
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw
from src.common.image_io import image_paths, load_rgb
from src.common.logging_utils import setup_logging

def organize(faces: pd.DataFrame, assignments: pd.DataFrame, source: Path, output: Path,
             images: pd.DataFrame | None = None, link: str = "copy", dry_run: bool = False,
             folder_names: dict[int, str] | None = None):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the source photo directory")
    if link not in {"copy", "hard", "symlink"}: raise ValueError("Invalid link mode")
    if faces.face_id.duplicated().any() or assignments.face_id.duplicated().any():
        raise ValueError("Duplicate face IDs")
    if set(faces.face_id) != set(assignments.face_id): raise ValueError("Assignments do not match faces")
    merged = faces.merge(assignments, on="face_id", validate="one_to_one")
    counts = merged[merged.cluster_id >= 0].groupby("cluster_id").size()
    ordered = sorted(counts.index, key=lambda i: (-counts[i], i))
    folder_map = {i: f"Person_{n:02d}" for n, i in enumerate(ordered, 1)}
    if folder_names:
        folder_map.update({i: folder_names[i] for i in ordered if i in folder_names})
    names = list(folder_map.values())
    if any(not n or n in {".", ".."} or any(c in n for c in '/\\:\x00') or n.casefold() in {"unknown", "no_face", "manifest.csv", "summary.json"} for n in names):
        raise ValueError("Invalid person folder name")
    if len({n.casefold() for n in names}) != len(names):
        raise ValueError("Person folder names must be unique")
    folder_map[-1] = "Unknown"
    destinations: dict[Path, set[str]] = {}
    for row in merged.itertuples():
        path = Path(row.image_path).resolve()
        if source not in path.parents: raise ValueError(f"Photo outside source: {path}")
        destinations.setdefault(path, set()).add(folder_map[row.cluster_id])
    unreadable = 0
    if images is not None:
        for row in images.itertuples():
            path = Path(row.image_path).resolve()
            if source not in path.parents: raise ValueError("Image inventory outside source")
            if row.status != "ok":
                unreadable += 1
                continue
            if path not in destinations:
                # Detected faces rejected by quality filters are Unknown, not No_Face.
                destinations[path] = {"No_Face" if row.detected == 0 else "Unknown"}
    else:
        # Without detector inventory, absence of accepted faces is ambiguous.
        missing = set(p.resolve() for p in image_paths(source)) - set(destinations)
        if missing: raise ValueError("Supply --images inventory to distinguish No_Face from filtered faces")
    planned, used = [], set()
    for path in sorted(destinations):
        for folder in sorted(destinations[path]):
            name = path.name
            if name.casefold() == "_preview.jpg" or (folder, name.casefold()) in used:
                name = hashlib.sha256(str(path).encode()).hexdigest()[:12] + "_" + name
            while (folder, name.casefold()) in used:
                name = "_" + name
            used.add((folder, name.casefold()))
            planned.append({"image": str(path), "folder": folder, "filename": name})
    manifest = pd.DataFrame(planned, columns=["image", "folder", "filename"])
    folder_counts = manifest.groupby("folder").size().to_dict() if len(manifest) else {}
    summary = {"source_images": len(destinations) + unreadable, "faces": len(faces),
               "persons": len(ordered), "copies": len(manifest), "counts_per_folder": folder_counts,
               "unknown": int(folder_counts.get("Unknown", 0)),
               "no_face": int(folder_counts.get("No_Face", 0)), "unreadable": unreadable,
               "link": link, "dry_run": dry_run}
    if not dry_run:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Output is not empty; choose a new directory to avoid stale photos")
        output.mkdir(parents=True, exist_ok=True)
        for row in manifest.itertuples():
            dst = output / row.folder / row.filename
            dst.parent.mkdir(exist_ok=True)
            if link == "copy": shutil.copy2(row.image, dst)
            elif link == "hard": os.link(row.image, dst)
            else: dst.symlink_to(row.image)
        manifest.to_csv(output / "manifest.csv", index=False)
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        for cluster_id, group in merged.groupby("cluster_id"):
            folder = output / folder_map[cluster_id]
            if not folder.exists() or "crop_path" not in group: continue
            sheet = Image.new("RGB", (4 * 112, 2 * 132), "white")
            draw = ImageDraw.Draw(sheet)
            for i, row in enumerate(group.head(8).itertuples()):
                crop = load_rgb(Path(row.crop_path)); crop.thumbnail((112, 112))
                left, top = (i % 4) * 112, (i // 4) * 132
                sheet.paste(crop, (left, top)); draw.text((left, top + 112), row.face_id[:10], fill="black")
            sheet.save(folder / "_preview.jpg")
    logging.info("Organizer summary: %s", json.dumps(summary))
    return manifest, summary

def main():
    p = argparse.ArgumentParser(description="Copy each original into every applicable person folder")
    p.add_argument("--faces", type=Path, required=True)
    p.add_argument("--assignments", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--images", type=Path, help="Detector images.csv inventory (required for zero-face photos)")
    p.add_argument("--link", choices=["copy", "hard", "symlink"], default="copy")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(); setup_logging("organize")
    organize(pd.read_parquet(a.faces), pd.read_csv(a.assignments), a.source, a.output,
             pd.read_csv(a.images) if a.images else None, a.link, a.dry_run)

if __name__ == "__main__": main()
