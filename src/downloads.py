"""Disk-backed ZIP downloads for the local app, including multi-GB photo sets."""
from __future__ import annotations
import hashlib
import json
import os
import secrets
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote
from src.product import ROOT, export_project, locked


def prepare_zip(project: Path, selected=None, progress=None):
    # Reuse the same reviewed manifest as ordinary folder export, without copying photos.
    manifest, summary = export_project(project, ROOT/'outputs/download_preview', selected, dry_run=True)
    if selected == [] or not len(manifest):
        raise ValueError('กรุณาเลือกบุคคลที่ต้องการดาวน์โหลด')
    fingerprint = {'revision':summary['review_revision'],
                   'files':[(r.image,r.folder,r.filename,Path(r.image).stat().st_size,Path(r.image).stat().st_mtime_ns)
                            for r in manifest.itertuples()]}
    key = hashlib.sha256(json.dumps(fingerprint,ensure_ascii=False).encode()).hexdigest()[:24]
    cache = project/'downloads'
    cache.mkdir(exist_ok=True)
    path = cache/f'{key}.zip'
    with locked(cache/'.build.lock'):
        if path.is_file():
            return path, summary
        if summary['required_bytes'] + 512*2**20 > summary['free_bytes']:
            raise ValueError('พื้นที่ไม่พอสำหรับเตรียม ZIP กรุณาเลือกดาวน์โหลดบางคนก่อน')
        temp = cache/f'{key}.partial'
        prefix = project.name
        try:
            with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
                for number,row in enumerate(manifest.itertuples(),1):
                    archive.write(row.image,f'{prefix}/{row.folder}/{row.filename}')
                    if progress:
                        progress(number,len(manifest))
                archive.writestr(f'{prefix}/manifest.csv',manifest.to_csv(index=False))
                archive.writestr(f'{prefix}/summary.json',json.dumps({**summary,'dry_run':False,'format':'zip'},ensure_ascii=False,indent=2))
                archive.writestr(f'{prefix}/export_info.json',json.dumps({'review_revision':summary['review_revision'],
                                       'selected_clusters':selected,'pretrained':True},ensure_ascii=False,indent=2))
            os.replace(temp,path)
        finally:
            temp.unlink(missing_ok=True)
    return path, summary


class LocalDownloads:
    """Expose only registered archives, by unguessable token, on loopback."""
    def __init__(self):
        self.files = {}
        self.lock = threading.Lock()
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):
                pass

            def do_HEAD(self):
                self.serve(False)

            def do_GET(self):
                self.serve(True)

            def serve(self,body):
                with owner.lock:
                    entry = owner.files.get(self.path.split('?',1)[0])
                if entry is None:
                    self.send_error(404); return
                path,name = entry
                try:
                    handle=path.open('rb')
                except OSError:
                    self.send_error(404); return
                with handle:
                    size=os.fstat(handle.fileno()).st_size
                    start,end=0,size-1
                    range_value=self.headers.get('Range')
                    if range_value:
                        try:
                            unit,bounds=range_value.split('=',1)
                            first,last=bounds.split('-',1)
                            if unit!='bytes' or ',' in bounds:
                                raise ValueError()
                            if first:
                                start=int(first); end=min(int(last),end) if last else end
                            else:
                                suffix=int(last)
                                if suffix<=0: raise ValueError()
                                start=max(0,size-suffix)
                            if start<0 or start>end or start>=size:
                                raise ValueError()
                        except ValueError:
                            self.send_response(416)
                            self.send_header('Content-Range',f'bytes */{size}')
                            self.end_headers(); return
                    self.send_response(206 if range_value else 200)
                    self.send_header('Content-Type','application/zip')
                    self.send_header('Content-Disposition',f"attachment; filename=photos.zip; filename*=UTF-8''{quote(name,safe='')}")
                    self.send_header('Content-Length',str(end-start+1))
                    self.send_header('Accept-Ranges','bytes')
                    self.send_header('Cache-Control','no-store')
                    self.send_header('X-Content-Type-Options','nosniff')
                    if range_value:
                        self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
                    self.end_headers()
                    if body:
                        handle.seek(start)
                        remaining=end-start+1
                        try:
                            while remaining:
                                data=handle.read(min(1024*1024,remaining))
                                if not data: break
                                self.wfile.write(data); remaining-=len(data)
                        except (BrokenPipeError,ConnectionResetError):
                            pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()

    def url(self,path: Path,name: str):
        route='/'+secrets.token_urlsafe(32)
        with self.lock:
            self.files[route]=(path.resolve(),name)
        return f'http://127.0.0.1:{self.server.server_port}{route}'

    def close(self):
        self.server.shutdown(); self.server.server_close()
