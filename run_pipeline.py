"""One-command local pretrained photo organizer; reviewed export is optional."""
import argparse
import json
from pathlib import Path
from src.product import project_path, run_project, export_project, write_json
from src.common.logging_utils import setup_logging


def main():
    p = argparse.ArgumentParser(description='จัดกลุ่มรูปในเครื่องด้วยโมเดลสำเร็จรูป')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--run-name', default='event')
    p.add_argument('--output', type=Path, help='คัดลอกรูปออกเป็นโฟลเดอร์บุคคล หากไม่ระบุจะสร้างชุดรูปเพื่อตรวจกลุ่มก่อน')
    p.add_argument('--algo', choices=['hdbscan', 'dbscan', 'agglomerative'], default='dbscan')
    p.add_argument('--threshold', type=float, default=.4)
    p.add_argument('--rescue-sim', type=float, default=None)
    p.add_argument('--no-rescue', action='store_true')
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    if a.limit is not None and a.limit < 1:
        p.error('limit must be positive')
    setup_logging('product_pipeline')
    project = project_path(a.run_name)
    try:
        summary = run_project(a.input, project, a.algo, a.threshold, None if a.no_rescue else a.rescue_sim, a.limit)
    except BlockingIOError:
        raise RuntimeError('ชุดรูปนี้กำลังประมวลผลอยู่แล้ว')
    except Exception as error:
        write_json(project/'status.json', {'state':'failed','phase':'ประมวลผลไม่สำเร็จ','error':str(error)})
        raise
    if a.output:
        _, summary = export_project(project, a.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
