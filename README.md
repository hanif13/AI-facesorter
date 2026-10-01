# face-sorter

โครงงานจัดรูปตามบุคคลสำหรับส่งวันที่ 7 ตุลาคม 2026 งานปัจจุบันคือ Phase 0–1 และจะหยุดรายงานที่ Gate 1 ก่อนเริ่ม Phase 2

Phase 0 และ Phase 1 ทดสอบแล้วบนรูปงานจริง 100 รูป อ่านผลและข้อจำกัดใน [report/gate1.md](report/gate1.md) ยังไม่เริ่ม training

**SCRFD detector เป็น pretrained component ที่อนุญาตไว้** ส่วน `w600k_r50.onnx` เป็น pretrained baseline สำหรับเปรียบเทียบเท่านั้น ยังไม่มี embedding model ที่เทรนเองในโครงการนี้

## สภาพแวดล้อม

ใช้ Python 3.11, macOS Apple Silicon และ PyTorch MPS สำหรับการเทรนในเฟสถัดไป ส่วน ONNX baseline ทำงานบน CPU

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install setuptools wheel cython cmake numpy
pip install --no-build-isolation insightface==0.7.3
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -m src.common.device
python -m scripts.check_environment
```

`requirements.in` คือรายการ dependency ที่ต้องการ ส่วน `requirements.txt` จะบันทึกเวอร์ชันที่ติดตั้งและตรวจสอบแล้ว

## Baseline pipeline

รันจากโฟลเดอร์โครงการ โดยใช้โฟลเดอร์ผลลัพธ์ที่ว่าง การรันตรวจจับครั้งแรกดาวน์โหลด buffalo_l เก็บภายใน outputs/models

```sh
python -m src.detect_align --input /path/to/photos --output outputs/embeddings/baseline
python -m src.baseline_pretrained --faces outputs/embeddings/baseline/faces.parquet
python -m src.cluster --embeddings outputs/embeddings/baseline/embeddings.npy --face-ids outputs/embeddings/baseline/embedding_faces.csv
python -m src.organize --faces outputs/embeddings/baseline/faces.parquet --assignments outputs/clusters/baseline/assignments.csv --images outputs/embeddings/baseline/images.csv --source /path/to/photos --output outputs/sorted_photos/baseline
python -m unittest discover -s tests -v
```

รองรับ `--algo dbscan|agglomerative|hdbscan`, `--no-rescue`, quality filters ใน detector และ `--dry-run` ใน organizer ค่า default เป็นจุดเริ่มต้น ยังไม่ได้ tune หรือวัด accuracy

สำหรับทดสอบรวดเร็ว (ใช้ชื่อ run ใหม่ทุกครั้ง):

```sh
python -m scripts.smoke_baseline --input /path/to/photos --limit 100 --run-name smoke100
```

รูปหลายคนถูกคัดลอกเข้าทุกโฟลเดอร์ที่เกี่ยวข้อง โดยไม่ซ้ำในโฟลเดอร์เดียวกัน รูปที่ตรวจไม่พบใบหน้าไป No_Face รูปที่ใบหน้าถูกตัดออกทั้งหมดเพราะ quality filter ไป Unknown รูปอ่านไม่ได้บันทึกใน images.csv และข้าม

ผลลัพธ์มี Person_XX, Unknown, No_Face, manifest.csv, summary.json และ contact sheet `_preview.jpg` ต่อกลุ่ม ต้นฉบับไม่ถูกแก้ไข โหมด hard link ต้องระวังว่าการแก้ไขไฟล์ผลลัพธ์จะกระทบต้นฉบับได้ แนะนำ copy ซึ่งเป็น default

## ข้อมูลและความเป็นส่วนตัว

ไม่อัปโหลดรูปงานหรือ embedding ออกนอกเครื่อง data/, outputs/, โมเดล และ virtual environment ถูกกันออกจาก git ให้ผู้จัดงานยืนยันการใช้รูปและการให้ความยินยอมก่อนนำไปใช้จริง

CASIA-WebFace aligned 112×112: [InsightFace dataset zoo](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) / [Google Drive](https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view)
หากดาวน์โหลดต้อง login ให้ดาวน์โหลดเองลง data/casia_raw และแจ้ง path ยังไม่ทำ Phase 2 data prep หรือ training ก่อน Gate 1
Mirror ทางเลือก: [Kaggle WebFace 112x112](https://www.kaggle.com/datasets/yakhyokhuja/webface-112x112)

## ข้อจำกัดปัจจุบัน

จำนวน cluster ไม่ใช่จำนวนบุคคลจริงที่ยืนยันแล้ว ต้องตรวจ contact sheets การ rescue เพิ่ม recall ได้แต่เพิ่ม false merges ได้เช่นกัน ไม่มี ground truth จึงยังรายงาน precision/recall ไม่ได้
