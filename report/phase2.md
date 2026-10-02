# Phase 2 — เตรียมข้อมูลฝึกแล้ว

รับ archive.zip จากผู้ใช้ รูปแบบ identity-folder aligned RGB 112×112

- รูปทั้งหมด 490,623 รูป / 10,572 identities
- Training 490,423 รูป / Validation 200 รูป จาก training identities สำหรับ sanity เท่านั้น
- รูปต่อ identity ตั้งแต่ 2 ถึง 802
- memmap uint8 RGB ใช้พื้นที่ 18,463,124,864 bytes
- อ่าน ZIP โดยตรง ตรวจ CRC ทุกภาพ ไม่ต้องแตก ZIP
- สุ่มตรวจ 64 รูปตรงกับ archive และตรวจ split ไม่ซ้ำ ทุก class ยังอยู่ใน training
- DataLoader workers=6, batch=128, augmentation เปิด, shuffle=True, persistent_workers=True, pin_memory=False
- ผ่านเต็ม epoch 3,831 batches / 490,368 รูป ใน 82.27 วินาที (drop_last=55 รูป)
- ความเร็วรวม startup 5960.60 img/s และหลัง warm-up 6201.77 img/s ผ่านเป้าหมาย 3,000 img/s

หลักฐาน: data/casia_npy/dataset_stats.json, outputs/logs/data_integrity.json, outputs/logs/data_benchmark.json, outputs/logs/data_benchmark_host_console.log

ตรวจภาพ outputs/report_assets/aug_samples.png แล้วใบหน้ายังอ่านได้ มี flip, brightness/contrast, crop และ erasing ตามแผน

ไฟล์นี้ไม่มี LFW ต้องได้ชุด evaluation เพิ่ม ยังไม่มีการเทรน embedding หรืออ้าง accuracy

Sandbox บล็อก torch_shm_manager จึงทดสอบ DataLoader บนเครื่องจริง ระบบอนุมัติอัตโนมัติครั้งแรก timeout แต่ retry สำเร็จและ full epoch จบด้วย exit code 0
