# Gate 1 — ผล Phase 0 และ Phase 1

วันที่ 1 ตุลาคม 2026 — หยุดที่ Gate 1 ตามแผน ยังไม่เริ่ม data preparation หรือ training

## สภาพแวดล้อมที่ตรวจสอบแล้ว

- RAM 16 GB ตามข้อมูลจากผู้ใช้และการตรวจฮาร์ดแวร์
- Python 3.11.16 / macOS Apple Silicon
- MPS available=True และ built=True พร้อมทดสอบ matrix multiplication สำเร็จ โปรแกรมจบด้วย exit code 0
- imports ที่ต้องใช้ผ่าน และบันทึกเวอร์ชันจริงใน requirements.txt

หลักฐาน: `outputs/logs/environment_verified.json`, `outputs/logs/environment_verified.log`, `outputs/logs/mps_standalone.log`

ข้อควรระวัง: sandbox มองไม่เห็น MPS ต้องรันจาก terminal บนเครื่องจริง การตรวจ imports รวม ONNX/OpenCV และคำนวณ MPS ใน process เดียวเคยคำนวณสำเร็จแต่ abort ตอนปิดโปรแกรม จึงแยก process สำหรับการตรวจสอบไว้แล้ว ยังไม่ได้ตรวจ training ซึ่งจะทำในเฟสถัดไป หลักฐานเดิมเก็บใน `outputs/logs/environment_host_check.log`

## Baseline smoke test บนรูปงานจริง

ใช้ 100 รูปแรกตามการเรียง path จากโฟลเดอร์ที่ผู้ใช้ระบุ ไม่ได้สุ่มให้เป็นตัวแทนทั้งงาน

| รายการ | ผลที่บันทึกจริง |
|---|---:|
| รูปที่ประมวลผล | 100 |
| ใบหน้าที่ผ่าน quality filters | 426 |
| กลุ่มที่ HDBSCAN สร้าง | 106 |
| รายการคัดลอกรูปเข้าทุกโฟลเดอร์ | 413 |
| รูปที่อยู่มากกว่าหนึ่งโฟลเดอร์ | 70 |
| รูปใน Unknown | 21 |
| รูปใน No_Face | 2 |
| รูปอ่านไม่ได้ | 0 |
| ใบหน้าที่เหลือเป็น noise | 32 |
| ใบหน้าที่ถูก noise rescue | 70 |
| เวลารันรวม (วินาที รวม download/startup) | 160.21 |

หลักฐาน: `outputs/embeddings/smoke100/run_summary.json`, `outputs/logs/smoke100_integrity.json`, `outputs/clusters/smoke100/assignments.csv`, `outputs/sorted_photos/smoke100/manifest.csv`, `outputs/logs/smoke100_console.log`

**106 กลุ่มไม่ใช่จำนวนบุคคลจริงที่ยืนยันแล้ว** ยังไม่มี ground truth และยังไม่ได้ tune clustering จึงไม่รายงาน accuracy, precision, recall หรือ F1 ภาพ preview แสดงใบหน้ามุมข้าง เบลอ และบาง crop ที่ควรให้คนตรวจ ไม่ควรถือว่าการจัดกลุ่มทุกกลุ่มถูกต้อง

ภาพ preview: `outputs/report_assets/baseline_smoke100_preview.jpg` และ `_preview.jpg` ในแต่ละ Person folder

## การตรวจสอบความถูกต้องของไฟล์

- unit tests 4 ข้อผ่าน รวมรูปเดียวเข้าทั้งสอง Person folders และไม่ซ้ำใน folder เดียว, ชื่อไฟล์ชนกัน, zero-face/filtered-face routing, empty/singleton embeddings และกลุ่มสังเคราะห์
- ตรวจ SHA-256 ของไฟล์คัดลอกทุกไฟล์เทียบกับต้นฉบับ และตรวจ manifest กับ assignments
- ใช้ copy mode ต้นฉบับไม่ได้ถูกย้ายหรือแก้ไข รูป, crops, embeddings, outputs และ model weights ไม่ถูก commit เข้า git

หลักฐาน: `outputs/logs/baseline_unit_tests.log`, `outputs/logs/smoke100_integrity.json`

## สถานะ CASIA-WebFace

ลิงก์ Google Drive ใน [InsightFace dataset zoo](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) ตอบ **Quota exceeded** จึงยังไม่ได้ไฟล์ CASIA และไม่มีการดาวน์โหลดชุดข้อมูลที่กำลังทำงาน

หลักฐาน: `outputs/logs/casia_download_response.html` — curl จบโดย HTTP สำเร็จ แต่ตรวจ signature แล้วพบว่าเป็น HTML ไม่ใช่ ZIP จึงไม่เก็บเป็น archive ปลอม

ลอง Kaggle public endpoint แล้วได้ HTTP 404 หลักฐานใน `outputs/logs/kaggle_download_probe.log` ไม่พบ ~/.kaggle/kaggle.json จึงต้องให้ผู้ใช้ดาวน์โหลดเองหรือระบุไฟล์ที่มีอยู่แล้ว

ทางเลือกให้ผู้ใช้ตรวจ: [Kaggle WebFace 112x112](https://www.kaggle.com/datasets/yakhyokhuja/webface-112x112) ดาวน์โหลดลง `data/casia_raw/` แล้วแจ้ง path ข้อมูลต้องเป็นรูปพร้อม identity labels หรือ train.rec/train.idx ไม่ใช้ pretrained embeddings แทนรูปฝึก

## ขอบเขต model ที่ใช้

**SCRFD detector จาก buffalo_l เป็น pretrained component ที่อนุญาตตามโจทย์**

**w600k_r50.onnx recognizer เป็น pretrained baseline เท่านั้น** ผลลัพธ์ระบุ embedder=pretrained_baseline ชัดเจน ไม่ใช้เป็น own model ยังไม่ได้สร้างหรือเทรน MobileFaceNet

## สิ่งที่ต้องการก่อนเฟสต่อไป

- ผู้ใช้ตรวจ preview และรับทราบผล Gate 1
- path ของ CASIA-WebFace หรือชุดข้อมูลทางเลือกที่มี identity labels
- ก่อนเริ่ม long training ต้องผ่าน benchmark, overfit test, resume test และหยุดขออนุมัติที่ Gate 2 ตามแผน
