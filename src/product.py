"""Local photo projects, review edits and safe export for the pretrained product."""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd
from src.common.image_io import image_paths
from src.organize import organize

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / 'outputs/projects'
DEFAULT_SOURCE = '/Users/hanif/Guidance For Dreams/Guidance For Dreams Season 6/รูปภาพ/day 3'


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        temp = f.name
    os.replace(temp, path)


@contextlib.contextmanager
def locked(path: Path, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def project_path(name: str) -> Path:
    if not name.strip() or Path(name).name != name or name in {'.', '..'} or '\\' in name:
        raise ValueError('ชื่อชุดรูปต้องเป็นชื่อเดียว ไม่มีเครื่องหมาย / หรือ \\')
    path = PROJECTS / name
    if path.resolve().parent != PROJECTS.resolve():
        raise ValueError('ตำแหน่งชุดรูปไม่ถูกต้อง')
    return path


def catalog(project: Path):
    config = read_json(project / 'project.json')
    if not config:
        raise ValueError('ไม่พบข้อมูลชุดรูป')
    work = Path(config.get('work', str(project / 'work')))
    faces = pd.read_parquet(work / 'faces.parquet')
    original = pd.read_csv(project / 'assignments.csv')
    review = read_json(project / 'review.json', {'revision': 0, 'names': {}, 'overrides': {}})
    assignments = original.copy()
    assignments['cluster_id'] = [review['overrides'].get(str(f), int(c)) for f, c in zip(assignments.face_id, assignments.cluster_id)]
    merged = faces.merge(assignments, on='face_id', validate='one_to_one')
    counts = merged[merged.cluster_id >= 0].groupby('cluster_id').size()
    ids = sorted(counts.index, key=lambda i: (-counts[i], i))
    names = {int(i): review['names'].get(str(i), f'Person_ID_{int(i)+1:03d}') for i in ids}
    names[-1] = 'ยังไม่ทราบบุคคล'
    return config, faces, assignments, merged, names, review


def edit_project(project: Path, action: str, **values):
    with locked(project / '.review.lock'):
        config, faces, assignments, merged, names, state = catalog(project)
        before = json.loads(json.dumps(state))
        valid = set(assignments.face_id.astype(str))
        if action == 'rename':
            cluster = int(values['cluster'])
            name = values['name'].strip()
            if cluster < 0 or cluster not in names:
                raise ValueError('เลือกกลุ่มบุคคลก่อน')
            if not name or name in {'.', '..'} or any(c in name for c in '/\\:\x00') or name.casefold() in {'unknown', 'no_face', 'manifest.csv', 'summary.json'}:
                raise ValueError('ชื่อไม่ถูกต้อง กรุณาใช้ชื่อบุคคลโดยไม่มี / หรือ \\')
            if any(n.casefold() == name.casefold() for i, n in names.items() if i != cluster):
                raise ValueError('ชื่อนี้มีอยู่แล้ว ใช้รวมกลุ่มหากเป็นคนเดียวกัน')
            state['names'][str(cluster)] = name
        elif action == 'merge':
            source, target = int(values['source']), int(values['target'])
            if source == target or source < 0 or target < 0 or source not in names or target not in names:
                raise ValueError('เลือกสองกลุ่มบุคคลที่ต่างกัน')
            for face in assignments.loc[assignments.cluster_id == source, 'face_id']:
                state['overrides'][str(face)] = target
        elif action == 'move':
            selected = [str(f) for f in values['faces']]
            if not selected or not set(selected) <= valid:
                raise ValueError('เลือกใบหน้าที่มีอยู่ในชุดรูป')
            target = values['target']
            if target == 'new':
                target = max([int(i) for i in names if i >= 0] + list(state['overrides'].values()) + [-1]) + 1
            target = int(target)
            if target != -1 and target not in names and values['target'] != 'new':
                raise ValueError('ไม่พบกลุ่มปลายทาง')
            for face in selected:
                state['overrides'][face] = target
        elif action == 'undo':
            previous = read_json(project / 'review_previous.json')
            if previous is None:
                raise ValueError('ยังไม่มีการแก้ไขให้ย้อนกลับ')
            state = previous
        else:
            raise ValueError('Unknown edit')
        state['revision'] = before.get('revision', 0) + 1
        write_json(project / 'review_previous.json', before)
        write_json(project / 'review.json', state)
        return state


def export_project(project: Path, output: Path, selected: list[int] | None = None, dry_run=False):
    config, faces, assignments, merged, names, state = catalog(project)
    work = Path(config.get('work', str(project / 'work')))
    inventory = pd.read_csv(work / 'images.csv')
    if selected is not None:
        keep = assignments.cluster_id.isin(selected)
        assignments = assignments[keep].copy()
        faces = faces[faces.face_id.isin(assignments.face_id)].copy()
        # Selected-person export contains only photos belonging to these people.
        inventory = inventory[inventory.image_path.isin(faces.image_path)]
    output = output.expanduser().resolve()
    if output == project.resolve() or project.resolve() in output.parents:
        raise ValueError('กรุณาส่งออกนอกโฟลเดอร์ข้อมูลของชุดรูปนี้')
    manifest, summary = organize(faces, assignments, Path(config['source']), output,
                                 inventory, dry_run=True, folder_names={i:n for i,n in names.items() if i >= 0})
    required = sum(Path(row.image).stat().st_size for row in manifest.itertuples())
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    summary.update(required_bytes=required, free_bytes=free, review_revision=state['revision'])
    if dry_run:
        return manifest, summary
    if required + 512 * 2**20 > free:
        raise ValueError(f'พื้นที่ไม่พอ ต้องใช้ประมาณ {required/2**30:.2f} GB แต่เหลือ {free/2**30:.2f} GB กรุณาเลือกบางคนหรือส่งออกไปไดรฟ์อื่น')
    organize(faces, assignments, Path(config['source']), output, inventory,
             folder_names={i:n for i,n in names.items() if i >= 0})
    summary['dry_run'] = False
    write_json(output / 'summary.json', summary)
    write_json(output / 'export_info.json', {'project': str(project), 'review_revision': state['revision'],
                                            'selected_clusters': selected, 'pretrained': True})
    return manifest, summary


def import_sample(name='ตัวอย่าง 100 รูป'):
    project = project_path(name)
    if (project / 'project.json').exists():
        return project
    work = ROOT / 'outputs/embeddings/smoke100'
    if not (work / 'faces.parquet').exists():
        return None
    project.mkdir(parents=True, exist_ok=True)
    detection = read_json(work / 'detection_config.json')
    write_json(project / 'project.json', {'source': detection['input'], 'work': str(work), 'sample_limit': 100,
               'embedder': 'pretrained InsightFace ArcFace R50', 'detector': 'pretrained SCRFD', 'imported_sample': True})
    shutil.copy2(ROOT / 'outputs/clusters/smoke100/assignments.csv', project / 'assignments.csv')
    write_json(project / 'review.json', {'revision': 0, 'names': {}, 'overrides': {}})
    write_json(project / 'status.json', {'state': 'ready', 'phase': 'พร้อมตรวจกลุ่ม', 'done': 100, 'total': 100})
    return project


def split_photo_conflicts(vectors, labels, faces, similarity=.6):
    """For ordinary event photos, two detections in one photo should not be one person.

    Only conflicting automatic groups are split; manual reviews remain authoritative.
    Collages/mirrors are a documented exception to this heuristic.
    """
    labels = np.asarray(labels).copy()
    next_id = int(labels.max(initial=-1)) + 1
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    for group in sorted(set(labels)-{-1}):
        indices = np.flatnonzero(labels == group)
        subset = faces.iloc[indices]
        if not subset.image_path.duplicated().any():
            continue
        buckets = []
        order = sorted(indices, key=lambda i: -float(faces.iloc[i].get('det_score', 1)))
        for index in order:
            photo = faces.iloc[index].image_path
            choices = []
            for number, bucket in enumerate(buckets):
                center = bucket['sum'] / max(np.linalg.norm(bucket['sum']),1e-12)
                score = float(center @ vectors[index])
                if photo not in bucket['photos'] and score >= similarity:
                    choices.append((score, number))
            if choices:
                bucket = buckets[max(choices)[1]]
                bucket['indices'].append(index); bucket['photos'].add(photo); bucket['sum'] += vectors[index]
            else:
                buckets.append({'indices':[index],'photos':{photo},'sum':vectors[index].copy()})
        for number, bucket in enumerate(buckets):
            if len(bucket['indices']) == 1:
                label = -1
            elif number == 0:
                label = group
            else:
                label = next_id; next_id += 1
            labels[bucket['indices']] = label
    return labels


def run_project(source: Path, project: Path, algorithm='dbscan', threshold=.4, rescue=None, limit=None):
    from src.detect_align import detect
    from src.baseline_pretrained import embed
    from src.cluster import cluster_vectors
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise ValueError('ไม่พบโฟลเดอร์รูปต้นฉบับ')
    if source == project.resolve() or source in project.resolve().parents:
        raise ValueError('โฟลเดอร์ชุดรูปต้องอยู่นอกโฟลเดอร์ต้นฉบับ')
    paths = image_paths(source)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError('โฟลเดอร์นี้ไม่มีรูปที่รองรับ')
    project.mkdir(parents=True, exist_ok=True)
    with locked(project / '.pipeline.lock', blocking=False):
        started = time.monotonic()
        work = project / 'work'
        models = ROOT / 'outputs/models'
        model = models / 'models/buffalo_l/w600k_r50.onnx'
        detector = models / 'models/buffalo_l/det_10g.onnx'
        if not model.is_file() or not detector.is_file():
            write_json(project / 'status.json', {'state':'running','phase':'กำลังดาวน์โหลดโมเดลสำหรับเปิดใช้ครั้งแรก'})
            from insightface.utils.storage import ensure_available
            ensure_available('models','buffalo_l',root=str(models))
        fingerprint = hashlib.sha256(json.dumps([(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]).encode()).hexdigest()
        signature = {'source': str(source), 'inventory_sha256': fingerprint, 'sample_limit': limit,
                     'algorithm': algorithm, 'threshold': threshold, 'rescue_sim': rescue,
                     'det_score': .5, 'min_face': 40., 'min_blur': 0.,
                     'photo_conflict_split': True,
                     'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
                     'detector_sha256': hashlib.sha256(detector.read_bytes()).hexdigest(),
                     'embedder': 'pretrained InsightFace ArcFace R50', 'detector': 'pretrained SCRFD'}
        old = read_json(project / 'project.json')
        if old and old != signature:
            raise ValueError('ชุดรูปนี้มีต้นฉบับหรือการตั้งค่าต่างจากเดิม กรุณาใช้ชื่อชุดรูปใหม่')
        write_json(project / 'project.json', signature)
        def status(phase, done=0, total=0, state='running', **extra):
            write_json(project / 'status.json', {'state': state, 'phase': phase, 'done': done, 'total': total,
                                                'elapsed_seconds': time.monotonic()-started, **extra})
        try:
            if not (work / 'detection_complete.json').exists():
                status('กำลังค้นหาใบหน้า', 0, len(paths))
                detect(source, work, models, limit=limit,
                       progress=lambda d,t: status('กำลังค้นหาใบหน้า',d,t))
                write_json(work / 'detection_complete.json', {'ok': True})
            if not (work / 'embedding_complete.json').exists():
                status('กำลังเปรียบเทียบใบหน้า')
                embed(work / 'faces.parquet', model, work,
                      progress=lambda d,t: status('กำลังเปรียบเทียบใบหน้า',d,t))
                write_json(work / 'embedding_complete.json', {'ok': True})
            if not (project / 'assignments.csv').exists():
                status('กำลังจัดกลุ่มบุคคล')
                vectors = np.load(work / 'embeddings.npy', allow_pickle=False)
                labels, rescued = cluster_vectors(vectors, algorithm=algorithm, eps=threshold,
                                                    distance_threshold=threshold, rescue_sim=rescue)
                ids = pd.read_csv(work / 'embedding_faces.csv')
                aligned_faces = pd.read_parquet(work / 'faces.parquet').set_index('face_id').loc[ids.face_id].reset_index()
                previous = labels.copy()
                labels = split_photo_conflicts(vectors,labels,aligned_faces,1-threshold)
                rescued[labels != previous] = False
                ids.assign(cluster_id=labels, rescued=rescued).to_csv(project / 'assignments.csv', index=False)
            if not (project / 'review.json').exists():
                write_json(project / 'review.json', {'revision': 0, 'names': {}, 'overrides': {}})
            config, faces, assignments, merged, names, state = catalog(project)
            images = pd.read_csv(work / 'images.csv')
            summary = {'images': len(images), 'faces': len(faces), 'groups': len(names)-1,
                       'unknown_faces': int((assignments.cluster_id == -1).sum()),
                       'no_face_images': int(((images.status == 'ok') & (images.detected == 0)).sum()),
                       'unreadable_images': int((images.status != 'ok').sum()),
                       'runtime_seconds_this_run': time.monotonic()-started, 'pretrained': True,
                       'scope': 'automatic grouping; not measured identification accuracy'}
            write_json(project / 'summary.json', summary)
            status('พร้อมตรวจกลุ่ม', len(paths), len(paths), state='ready')
            return summary
        except Exception as error:
            status('ประมวลผลไม่สำเร็จ', state='failed', error=str(error))
            raise


def recluster_project(original: Path, destination: Path, algorithm='agglomerative', threshold=.45, rescue=None):
    """Try another grouping without rerunning models or overwriting reviewed results."""
    from src.cluster import cluster_vectors
    config, faces, assignments, merged, names, state = catalog(original)
    if destination.exists():
        raise ValueError('ชื่อชุดรูปใหม่นี้มีอยู่แล้ว กรุณาใช้ชื่ออื่น')
    work = Path(config.get('work', str(original/'work'))).resolve()
    vectors = np.load(work/'embeddings.npy',allow_pickle=False)
    ids = pd.read_csv(work/'embedding_faces.csv')
    if len(ids) != len(vectors):
        raise ValueError('ข้อมูลใบหน้าไม่ครบ')
    labels, rescued = cluster_vectors(vectors,algorithm=algorithm,eps=threshold,
                                      distance_threshold=threshold,rescue_sim=rescue)
    aligned_faces = faces.set_index('face_id').loc[ids.face_id].reset_index()
    previous = labels.copy()
    labels = split_photo_conflicts(vectors,labels,aligned_faces,1-threshold)
    rescued[labels != previous] = False
    destination.mkdir(parents=True)
    write_json(destination/'project.json',{**config,'work':str(work),'algorithm':algorithm,
                                         'threshold':threshold,'rescue_sim':rescue,'derived_from':str(original.resolve()),
                                         'photo_conflict_split':True})
    ids.assign(cluster_id=labels,rescued=rescued).to_csv(destination/'assignments.csv',index=False)
    write_json(destination/'review.json',{'revision':0,'names':{},'overrides':{}})
    images=pd.read_csv(work/'images.csv')
    summary={'images':len(images),'faces':len(faces),'groups':len(set(labels)-{-1}),
             'unknown_faces':int((labels==-1).sum()),'pretrained':True,
             'scope':'automatic grouping; not measured identification accuracy'}
    write_json(destination/'summary.json',summary)
    write_json(destination/'status.json',{'state':'ready','phase':'พร้อมตรวจกลุ่ม'})
    return summary
