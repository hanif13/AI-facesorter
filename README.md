# Face Sorter — จัดรูปงานตามบุคคล

ปรับข้อกำหนดวันที่ 3 ตุลาคม 2026 ตามที่ผู้ใช้ยืนยันกับอาจารย์: เน้นผลิตภัณฑ์ที่ใช้งานได้ อนุญาตใช้ไลบรารีและโมเดลสำเร็จรูป ไม่ต้องฝึก embedding จากศูนย์ ระบบหลักจึงใช้ SCRFD + ArcFace R50 ผ่าน InsightFace การฝึก 30 ชั่วโมงไม่อยู่ในงานที่จำเป็นอีกต่อไป

## เปิดโปรแกรม

ดับเบิลคลิก `เปิดโปรแกรม.command` หรือรันจากโฟลเดอร์ face-sorter:

```sh
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

เปิด http://127.0.0.1:8501 บนเครื่องนี้ เลือกชุดรูป `day3_ready` สำหรับรูป day 3 ทั้งหมด หรือ `ตัวอย่าง 100 รูป` สำหรับชุดทดลองเดิม

1. กด **เลือกโฟลเดอร์รูป** เพื่อเลือกจากหน้าต่าง Mac แล้วตั้งชื่อชุดรูปใหม่ ไม่ต้องพิมพ์ path
2. รอค้นหาและจัดกลุ่มใบหน้า ระบบแสดงความคืบหน้า
3. ดูใบหน้าหรือรูปงานในแต่ละกลุ่ม ตั้งชื่อ รวมกลุ่ม หรือเลือกใบหน้าเพื่อย้ายไปกลุ่มอื่น/สร้างกลุ่มใหม่/ยังไม่ทราบบุคคล การแก้ไขบันทึกทันทีและย้อนครั้งล่าสุดได้
4. เปิดแท็บ **ดาวน์โหลดรูป** เลือกทุกคนหรือเฉพาะบางคน กด **เตรียมไฟล์ ZIP สำหรับดาวน์โหลด** แล้วกด **ดาวน์โหลดโฟลเดอร์รูป (.zip)** ไม่ต้องกรอก path ปลายทาง แตก ZIP แล้วได้โฟลเดอร์บุคคลตามที่ตรวจแก้ไว้

ปุ่มเลือกโฟลเดอร์เปิดหน้าต่างบนเครื่องที่รันโปรแกรม สำหรับ Mac จะใช้ตัวเลือกโฟลเดอร์ของระบบ (สร้างจาก AppKit เมื่อมี Command Line Tools) และมี Tkinter เป็นทางเลือก ไม่คัดลอกหรืออัปโหลดรูปตอนเลือกโฟลเดอร์

โปรแกรมยังเปิดได้โดยไม่ต้องใช้ข้อมูลฝึก CASIA หรือ checkpoint ทดลอง ใช้เฉพาะโมเดล ONNX ใน outputs/models และรูปต้นฉบับ ข้อมูลใบหน้า กลุ่ม และการแก้ไขเก็บที่ outputs/projects/<ชื่อชุดรูป>/

## ใช้คำสั่งเดียว

สร้างชุดรูปเพื่อเปิดตรวจในหน้าจอ (ยังไม่คัดลอกรูป):

```sh
.venv/bin/python run_pipeline.py --input /path/to/photos --run-name my_event
```

จัดกลุ่มและคัดลอกออกทันที:

```sh
.venv/bin/python run_pipeline.py --input /path/to/photos --run-name my_event --output /path/to/sorted
```

รองรับ `--algo hdbscan|dbscan|agglomerative`, `--threshold`, `--rescue-sim`, `--no-rescue`, `--limit` ต้องใช้ชื่อชุดรูปใหม่เมื่อเปลี่ยนต้นฉบับหรือค่า การตั้งค่าและภาพเดิมเทียบด้วย path/size/mtime และ hash ของโมเดล ไม่ได้ hash เนื้อหารูปทั้งหมด เมื่อทำต่อจะใช้ขั้นตอนที่เสร็จแล้ว หากหยุดกลางขั้นตอนจะเริ่มขั้นตอนนั้นใหม่

หน้าเว็บรวมผลเป็น ZIP โดยอ่านรูปต้นฉบับโดยตรง ไม่สร้างสำเนาโฟลเดอร์ก่อนและไม่โหลด ZIP ทั้งก้อนเข้า RAM ไฟล์ ZIP ที่เตรียมไว้เก็บใน outputs/projects/<ชื่อชุดรูป>/downloads และใช้ซ้ำได้เมื่อผลตรวจแก้กับไฟล์ต้นฉบับไม่เปลี่ยน ต้องมีพื้นที่ว่างสำหรับ ZIP และไฟล์ที่เบราว์เซอร์ดาวน์โหลดอีกหนึ่งชุด การดาวน์โหลดใช้งานบนเครื่องเดียวกับที่รันโปรแกรม

ผลส่งออกเป็นโฟลเดอร์ชื่อบุคคล, Unknown, No_Face, manifest.csv, summary.json และ export_info.json รูปหลายคนคัดลอกเข้าทุกโฟลเดอร์ที่เกี่ยวข้อง ชื่อบุคคลที่แสดงก่อนผู้ใช้ตั้งเป็นรหัสกลุ่มอัตโนมัติ ไม่ใช่การระบุตัวตนจริง

รุ่นปัจจุบันเริ่มด้วย DBSCAN cosine distance 0.4 และปิด noise rescue ค่านี้ยังไม่ได้วัดความแม่นยำกับข้อมูลติดป้ายกำกับ กติกาช่วยแยกใบหน้าหลายคนจากรูปเดียวกันไม่เหมาะกับภาพ collage หรือกระจก ผู้ใช้แก้กลุ่มเองได้

## รายงานและการตรวจสอบ

- รายงานหลักตามข้อกำหนดใหม่: [report/report.md](report/report.md)
- แผนปัจจุบัน: [report/product_plan.md](report/product_plan.md)
- โครงสไลด์: [report/slides_outline.md](report/slides_outline.md)
- ผลทดสอบการคัดลอก การแก้กลุ่ม และข้อจำกัด: outputs/logs/product_unit_tests.log

```sh
.venv/bin/python -m unittest discover -s tests -v
```

โมเดลสำเร็จรูปของ InsightFace มีเงื่อนไข non-commercial research ตาม [เอกสารเจ้าของโมเดล](https://github.com/deepinsight/insightface/blob/master/python-package/README.md#license) รุ่นนี้มุ่งใช้เป็นโครงงานมหาวิทยาลัย ต้องตรวจสิทธิ์โมเดลก่อนนำไปใช้เชิงพาณิชย์ โค้ด MIT และสิทธิ์น้ำหนักโมเดลเป็นคนละส่วน

## ประวัติการทดลองก่อนเปลี่ยนข้อกำหนด

ส่วนด้านล่างเก็บวิธีติดตั้งและการทดลอง Phase 0–3 ไว้อ้างอิง คำสั่งฝึกโมเดลเป็นทางเลือกเชิงวิจัย ไม่จำเป็นสำหรับโปรแกรมรุ่นปัจจุบัน รายงาน gate1/gate2 เป็นบันทึกข้อกำหนดเดิม ไม่ใช่เงื่อนไขอนุมัติของผลิตภัณฑ์รุ่นใหม่

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

## Pipeline ทดลองเดิม

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
หากดาวน์โหลดต้อง login ให้ดาวน์โหลดเองลง data/casia_raw และแจ้ง path ขั้นตอนถัดไปเริ่มได้เมื่อผ่าน Gate 1 แล้ว
Mirror ทางเลือก: [Kaggle WebFace 112x112](https://www.kaggle.com/datasets/yakhyokhuja/webface-112x112)

## Phase 2: เตรียมข้อมูลฝึก

รองรับ ZIP ที่แยกโฟลเดอร์ตาม identity หรือโฟลเดอร์รูปแบบเดียวกัน อ่าน ZIP โดยตรง ไม่ต้องแตกไฟล์ และตรวจ CRC ระหว่างอ่าน ไม่ต้องใช้ MXNet

```sh
python -m src.data.prepare_casia --source /path/to/archive.zip --output data/casia_npy
python -m scripts.benchmark_data --data data/casia_npy --workers 6 --batch-size 128
```

ผลลัพธ์: images_112.npy (uint8 RGB memmap), labels.npy (int32), train_indices.npy, val_indices.npy, identities.json, image_index.tsv, prep_config.json และ dataset_stats.json

หากหยุดระหว่างเตรียมข้อมูล รันคำสั่งเดิมเพิ่ม `--resume` ตัวอ่านบันทึกตำแหน่งทุก 10,000 รูป หากมีไฟล์เสร็จแล้วจะไม่เขียนทับ ใช้ `--image-size 96`, `--max-identities 5000` และ `--max-per-identity 50` ได้เมื่อจำเป็น ทั้งหมดบันทึกไว้ใน config

DataLoader เปิด memmap แยกในแต่ละ worker ใช้ tensor augmentation: flip, brightness/contrast, crop/resize, erasing และ optional Gaussian blur ตรวจภาพตัวอย่างได้ที่ outputs/report_assets/aug_samples.png

ตัวอย่าง validation 200 รูปมาจาก training identities สำหรับตรวจ sanity เท่านั้น ไม่ใช่การวัด generalization ต้องใช้ LFW แยกต่างหาก ไม่มีการอ้างว่า CASIA และ LFW ไม่มี identity overlap โดยยังไม่ได้ตรวจ

LFW evaluation mirror: [AgeDB/CALFW/CPLFW/LFW aligned 112×112](https://www.kaggle.com/datasets/yakhyokhuja/agedb-30-calfw-cplfw-lfw-aligned-112x112) เก็บไว้ใน data/eval/ แยกจากข้อมูลฝึก

## Phase 3: ฝึก embedding จาก random initialization

MobileFaceNet 512 มิติ / ArcFace ไม่มี pretrained weights บันทึก initial state hash และ provenance ใน checkpoint ตัวฝึกใช้ MPS fp32 และมีเพดานหน่วยความจำสำหรับ Mac RAM 16 GB

คำสั่งเตรียม LFW จาก ZIP ที่มี BMP และ lfw_ann.txt:

```sh
python -m src.data.prepare_lfw --source data/eval/evaluation_download.zip
```

คำสั่งต่อไปนี้เป็นตัวอย่างสำหรับการฝึกเชิงวิจัยตามแผนเดิม (ไม่ได้เริ่มรัน และไม่จำเป็นตามข้อกำหนดใหม่):

```sh
export PYTORCH_ENABLE_MPS_FALLBACK=1
caffeinate -i .venv/bin/python -m src.train_embedding --config configs/train.yaml --time_budget_hours 30 --run-name casia_own_main
```

configuration เสนอใช้ 5,000 identities สูงสุด 50 รูปต่อคน จาก memmap เต็ม จำนวน epochs จะคำนวณจาก benchmark 200 iterations และ budget ไม่โหลด pretrained embedding โปรแกรมกัน overhead 10% และหยุดตาม wall-clock budget หรือเมื่อดิสก์ว่างต่ำกว่า 2 GiB

เก็บ initial.pt, checkpoint ทุก epoch, last.pt, best.pt ตาม LFW และ crash.pt แบบ atomic ภายใน outputs/checkpoints/<run-name>/ พร้อม config.yaml, iterations.csv, history.json และ log แบบ timestamp ใน outputs/logs/

```sh
python -m src.train_embedding --resume outputs/checkpoints/casia_own_main/last.pt
# หากจำเป็นต้องลด learning rate หลังวิเคราะห์ log:
python -m src.train_embedding --resume outputs/checkpoints/casia_own_main/last.pt --lr 0.025
```

SIGINT/Ctrl+C หยุดหลัง optimizer update จบ แล้วบันทึก crash.pt/last.pt Resume กู้ model/head/optimizer/scheduler และ progress ได้ แต่ augmentation RNG ภายใน workers ไม่ได้รับรอง bit-for-bit ระยะเวลาที่ใช้ก่อนหยุดถูกรวมใน budget เดิม ไม่เริ่ม budget 30 ชั่วโมงใหม่ทุกครั้งที่ resume

LFW ประเมินทุก 2 epochs ด้วย flip-TTA, L2 normalization และ 10-fold CV เลือก threshold เฉพาะอีก 9 folds ผล diagnostic มีป้ายแยกจาก trained model checkpoint gate2_* เป็นผลทดลองเดิม ผลิตภัณฑ์ปัจจุบันใช้ pretrained ONNX

```sh
python -m src.eval_lfw --bin data/eval/lfw.bin --embedder own --checkpoint outputs/checkpoints/casia_own_main/best.pt --output outputs/report_assets/lfw_own
python -m src.eval_lfw --bin data/eval/lfw.bin --embedder baseline --output outputs/report_assets/lfw_baseline
```

LFW ถูกใช้เลือก best checkpoint ตามแผน จึงควรระบุ model-selection bias ในรายงาน ผลบนรูปงานจริงต้องใช้ ground truth และ tune/test split แยกกันในเฟสต่อไป

## ข้อจำกัดปัจจุบัน

จำนวน cluster ไม่ใช่จำนวนบุคคลจริงที่ยืนยันแล้ว ต้องตรวจ contact sheets การ rescue เพิ่ม recall ได้แต่เพิ่ม false merges ได้เช่นกัน ไม่มี ground truth จึงยังรายงาน precision/recall ไม่ได้
# AI-facesorter
