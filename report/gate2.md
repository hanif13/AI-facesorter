# Gate 2 — พร้อมเสนอการเทรนยาว

วันที่ 2 ตุลาคม 2026 — หยุดที่ Gate 2 ตามแผน ยังไม่ได้เริ่ม Phase 4

## Model และหลักฐาน random initialization

MobileFaceNet ตาม architecture ในแผน, embedding 512 มิติ, 1,200,512 parameters ใช้ Kaiming initialization สำหรับ Conv และ constants สำหรับ BatchNorm

บันทึกข้อความ No pretrained weights loaded และ SHA-256 ของ state เริ่มต้น: `2d9c06312e6a016fc747b0d954a6ec8b687b4e4e1bec23ec37cf1761c308e635`

ArcFace s=64, target m=0.5, margin warm-up ระดับ epoch: 0, 0.25, 0.5 หลังผ่านสอง epochs แรก ทดสอบ zero-margin softmax, angular formula และกรณี angle เลย pi แล้ว

SCRFD เป็น pretrained detector ที่อนุญาต ส่วน pretrained recognizer อยู่ใน baseline เท่านั้น ไม่มี pretrained embedding weights ถูกโหลดเข้า MobileFaceNet

หลักฐาน: outputs/checkpoints/gate2_bounded_112_b128/benchmark.json, outputs/logs/gate2_bounded_112_b128.log

## Benchmark จริงบน MPS

| Image size | Batch | img/s | Driver memory sampled หลัง update | ผล |
|---|---:|---:|---:|---|
| 112 | 128 | 219.19 | 6.33 GiB | 300 iterations ผ่าน |
| 112 | 256 | ไม่รายงาน | ไม่รายงาน | MPS out of memory ภายใต้ cap 0.7 |
| 96 | 256 | ไม่รายงาน | ไม่รายงาน | MPS out of memory ภายใต้ cap 0.7 |

ใช้ full training split และ classifier 10,572 classes; warm-up 20 iterations แล้ววัด 280 iterations ส่วน memory เป็นค่าสุ่มวัดท้าย update รวม allocator/cache ไม่ใช่ peak activation ระหว่าง forward/backward

การลอง batch 256 ก่อนตั้ง cap ทำให้ RAM/swap สูง จึงหยุดการทดสอบนั้น แล้วทดสอบใหม่ด้วย `mps_memory_fraction=0.7` ทั้ง batch 256 ที่ 112 และ 96 ล้มเหลวจริง ไม่อ้าง throughput ของ configuration ที่ไม่สำเร็จ

เลือก 112×112, batch 128 ซึ่งผ่าน 300 iterations ภายใต้ cap

หลักฐาน: outputs/logs/training_benchmarks.json, outputs/logs/gate2_bounded_112_b128.log, outputs/logs/gate2_bounded_112_b256.log, outputs/logs/gate2_bounded_96_b256.log, outputs/logs/training_benchmarks_bounded_console.log

## Sanity tests

- Overfit: 50 identities × 20 images = 1,000 รูป, 200 optimizer updates, top-1 บนชุดฝึกย่อย = 100.0% ผ่านเกณฑ์ 90% **เป็นการทดสอบ memorization ไม่ใช่ accuracy บนคนใหม่**
- Mini-epoch: สุ่ม 5% ของ training split, 24,521 รูป, ประมวลผล 24,448 รูปใน 191 batches (drop_last=True); ค่าเฉลี่ย loss ของ 5 จุด log แรก 13.2206 → 5 จุดท้าย 12.6555 ลด 4.27% เป็นการลดเล็กน้อย ยังไม่ใช่หลักฐานว่าโมเดลใช้งานจริงได้ ไม่พบ NaN/Inf ใน run
- Resume: ส่ง SIGINT ให้ run ของเรา หยุดหลัง optimizer update จบที่ step 3 แล้ว resume จนถึง step 10; optimizer/scheduler และ provenance ถูกเก็บไว้ ทดสอบนี้ปิด augmentation และไม่ได้รับรองการต่อ RNG ของ worker แบบ bit-for-bit
- Unit tests 9 ข้อผ่าน รวมตรวจ threshold ของ fold ไม่ใช้ label ของ held-out fold

หลักฐาน: outputs/logs/overfit_result.json, outputs/checkpoints/gate2_overfit/history.json, outputs/checkpoints/gate2_mini/iterations.csv, outputs/logs/mini_epoch_summary.json, outputs/logs/resume_verified.json, outputs/logs/resume_boundary_console.log, outputs/logs/training_unit_tests_final.log

## LFW และ checkpoint integration

ดาวน์โหลด [evaluation mirror](https://www.kaggle.com/datasets/yakhyokhuja/agedb-30-calfw-cplfw-lfw-aligned-112x112) ได้แล้ว รูป BMP aligned และ annotation ถูกแปลงเป็น lossless PNG ใน local lfw.bin โดยรักษาลำดับคู่เดิม

ตรวจพบ 6,000 pairs (12,000 images), 3,000 same + 3,000 different; 10 contiguous folds ละ 600 คู่ มี same 300 คู่ทุก fold ยังไม่ได้เทียบ canonical image checksums อย่างอิสระ

รัน integration ครบ 6,000 pairs พร้อม flip-TTA, 10-fold threshold selection, ROC, และ best.pt บน diagnostic model ที่อัปเดตเพียง 2 ครั้ง จบด้วย exit code 0 ตัวเลขใน run นี้ติดป้าย diagnostic-only และ **ไม่ใช่ผล LFW ของโมเดลหลักที่เทรนเต็ม** ยังไม่มีผลคุณภาพของโมเดลหลักให้รายงาน

หลักฐาน: outputs/logs/lfw_preparation.json, outputs/checkpoints/gate2_lfw_integration/lfw_epoch2/metrics.json, outputs/checkpoints/gate2_lfw_integration/best.pt, outputs/logs/lfw_integration_console.log

## Configuration ที่เสนอให้อนุมัติ

- ใช้ 5,000 identities ที่มีรูปมากที่สุด สูงสุด 50 รูปต่อ identity: 218,208 รูป เลือกจาก training indices ไม่ใช้ validation/LFW เป็นข้อมูลฝึก
- 112×112 / embedding 512 / batch 128 / workers 6 / fp32 / SGD lr=0.05, momentum=0.9 / grad clip=5 / MPS fraction=0.7
- 1,704 batches ต่อ epoch; **ประมาณ** 16.58 นาทีต่อ epoch จาก benchmark classifier เต็ม จึงเป็น conservative estimate ไม่ใช่การวัดความเร็วของ subset จริง
- เสนองบ **30 ชั่วโมง wall-clock** ประมาณ **97 epochs** เมื่อกัน overhead 10% จำนวนจริงคำนวณใหม่จาก benchmark 200 iterations บน subset ตอนเริ่ม run ที่ได้รับอนุมัติ
- Cosine schedule, warm-up, checkpoint ทุก epoch, last.pt, best.pt ตาม LFW, initial.pt, crash.pt และ resume พร้อม explicit --lr override
- LFW ทุก 2 epochs, flip-TTA, evaluation batch 64; validation จาก training identities ใช้ดู loss เท่านั้น
- หยุดและเซฟ checkpoint เมื่อถึง budget หรือพื้นที่ว่างต่ำกว่า 2 GiB

หลักฐานของตัวเลขประมาณการและ selection hash: outputs/logs/proposed_training.json; configuration: configs/train.yaml

ก่อนเริ่ม ต้องเสียบปลั๊ก เปิดฝา/ใช้จอนอกตามแผน วางในที่ระบายอากาศ และใช้ caffeinate ต้องจบการเทรนภายในเย็นวันที่ 4 ตุลาคมตามแผนเดิม

**รอผู้ใช้อนุมัติ Gate 2 ก่อนเริ่มคำสั่งเทรนยาว**
