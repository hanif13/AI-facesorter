"""Open a native folder dialog in a separate main-thread Python process.

The local app reads photos in place; selecting a folder does not upload/copy it.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path


def native_binary() -> Path:
    root = Path(__file__).resolve().parents[1]
    source = root / 'src/native_folder_picker.swift'
    contents = root / 'tools/folder-picker/FaceSorterFolderPicker.app/Contents'
    binary = contents / 'MacOS/FaceSorterFolderPicker'
    stamp = contents / 'source.sha256'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if binary.exists() and stamp.exists() and stamp.read_text() == digest:
        return binary
    binary.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(['xcrun', 'swiftc', str(source), '-o', str(binary),
                             '-module-cache-path', str(root/'tools/folder-picker/module-cache')],
                            capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError('Cannot build native folder picker')
    with (contents/'Info.plist').open('wb') as handle:
        plistlib.dump({'CFBundleExecutable':'FaceSorterFolderPicker',
                      'CFBundleIdentifier':'local.facesorter.folderpicker',
                      'CFBundleName':'Face Sorter Folder Picker',
                      'CFBundlePackageType':'APPL', 'CFBundleVersion':'1'}, handle)
    stamp.write_text(digest)
    return binary


def choose_folder(initial: Path | None = None) -> Path | None:
    command = [sys.executable, '-m', 'src.folder_picker']
    if initial is not None:
        command += ['--initial', str(initial)]
    try:
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=300, check=False)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError('หน้าต่างเลือกโฟลเดอร์ปิดเพราะรอนาน กรุณากดเลือกใหม่') from error
    if result.returncode != 0:
        raise RuntimeError('เปิดหน้าต่างเลือกโฟลเดอร์ไม่ได้ กรุณาเปิดโปรแกรมบนเครื่อง Mac นี้แล้วลองอีกครั้ง')
    try:
        selected = json.loads(result.stdout)['folder']
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError('อ่านโฟลเดอร์ที่เลือกไม่ได้ กรุณาลองอีกครั้ง') from error
    if not selected:
        return None
    path = Path(selected).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError('ไม่พบโฟลเดอร์ที่เลือก กรุณาเลือกใหม่')
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--prepare-native', action='store_true')
    args = parser.parse_args()
    if sys.platform == 'darwin' and shutil.which('xcrun'):
        try:
            binary = native_binary()
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            if args.prepare_native:
                raise
        else:
            if args.prepare_native:
                print(binary)
                return
            os.execv(str(binary), [str(binary)] + ([str(args.initial)] if args.initial else []))
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    try:
        options = {'parent': root, 'title': 'เลือกโฟลเดอร์รูป — Face Sorter', 'mustexist': True}
        if args.initial and args.initial.is_dir():
            options['initialdir'] = str(args.initial)
        selected = filedialog.askdirectory(**options)
        print(json.dumps({'folder': selected or None}, ensure_ascii=False))
    finally:
        root.destroy()


if __name__ == '__main__':
    main()
