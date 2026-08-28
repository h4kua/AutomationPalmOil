# Deteksi Objek Kebun Sawit PSN — Dokumen Utama

**Sejarah lengkap eksperimen (versi apa aja yang dicoba, berhasil apa nggak, kenapa) ada di `report/PROGRESS_REPORT.md`.** Dokumen ini isinya status terkini + cara pakai.

**Soal path:**

- `data/...`, `scripts/...`, `report/...`, `models/...` → relatif ke folder `FINAL_DELIVERABLE/` ini sendiri.
- `runs/...`, `eval_sets/...` → satu tingkat di atas, folder project `psn-training/` (jejak bukti proses training/validasi, di luar isi deliverable ini).

## 1. Ringkasan Project

**Sistem deteksi objek buat pengawasan drone di kebun sawit, 2 model yang jalan sendiri-sendiri:**

- RGB (siang hari) — 6 kelas: Bridge, Car, Motorcycle, Palm_Oil_Fruit, Person, Truck
- Thermal/infrared (malam hari) — cuma Person & Car (nggak ada dataset thermal publik yang ada label Bridge atau Palm_Oil_Fruit)

Dua-duanya pakai YOLOv8. Yang dipakai sekarang: **RGB v12** dan **thermal v1**.

## 2. Jumlah Anotasi — dicek ulang tanggal 12 Agustus 2026

**4.934 task udah dianotasi, 2.643 objek total (6 kelas).** Diambil langsung dari database Label Studio (project 4, tabel `task` JOIN `task_completion`, server <LABEL_STUDIO_SERVER>) — bukan angka lama yang dipakai ulang.

|  | Jumlah |
| --- | --- |
| Total task yang udah dianotasi | 4.934 |
| Task yang ada minimal 1 objek dilabeli | 1.236 |
| Task yang dikonfirmasi kosong (background, nggak ada objek) | 3.698 |
| Total objek yang dilabeli (semua 6 kelas) | 2.643 |

**Per kelas:**

| Kelas          | Jumlah objek | Ada di berapa gambar |
| -------------- | ------------ | -------------------- |
| Bridge         | 752          | 513                  |
| Car            | 147          | 133                  |
| Motorcycle     | 128          | 106                  |
| Palm_Oil_Fruit | 1.263        | 279                  |
| Person         | 165          | 69                   |
| Truck          | 188          | 173                  |

- Dicek 2 cara (query database langsung vs file anotasi mentah yang beneran dipakai bikin v12) — hasilnya sama persis.
- Semua kotak yang dilabeli ada di dalam 1.236 task yang sama yang dipakai buat bikin model v12.
- 3.698 task sisanya itu submission kosong yang dikonfirmasi (ada gambarnya, memang nggak ada objek).
- Angka lengkap: `data/dataset_summary.json`.

## 3. Data dari Luar (Open-Source) — yang beneran dipakai vs yang cuma dicoba

**Training v12 yang jalan sekarang 100% nggak pakai data dari luar.** Sudah dicek langsung dari `scripts/prepare_dataset.py` — sumbernya cuma rekaman drone PSN sendiri yang dianotasi manual (`rgb_v9_images` / `rgb_v12_raw_annotations.txt`). Nggak ada jalur ke VisDrone, nggak ada data sintetis Kaggle, nggak ada dataset luar sama sekali di script itu.

- VisDrone dan data sintetis Kaggle **memang pernah** dicoba dan dites habis-habisan di eksperimen awal (`report/PROGRESS_REPORT.md` bagian 1.2).
- VisDrone dulu dikit bantu Car/Truck/Bridge, tapi malah bikin Person jelek — makanya nggak dipakai buat v12.
- Sengaja ditulis jelas di sini soalnya gampang salah sangka dari sejarah eksperimen itu kalau VisDrone kepake di model sekarang — padahal enggak.

## 4. Model yang Lagi Dipakai Sekarang

### Versi training RGB — 4 terbaik berdasarkan mAP50 di protected-59

**v12 rangking 2 dari 4 kalau dilihat mAP50 keseluruhan — tapi tetap dipakai, karena menang di 2 kelas paling penting buat deployment ini.**

| Ranking | Versi                      | mAP50 Keseluruhan | Bridge | Person | Palm_Oil_Fruit |
| ------- | -------------------------- | ----------------- | ------ | ------ | -------------- |
| 1       | v5 (awal)                  | 0,7792            | 0,4422 | 0,8656 | 0,8026         |
| 2       | **v12 (dipakai sekarang)** | 0,7626            | 0,5814 | 0,5324 | 0,8768         |
| 3       | v16                        | 0,7477            | 0,5374 | 0,2786 | 0,9200         |
| 4       | v13                        | 0,7363            | 0,4956 | 0,3108 | 0,9008         |

- Bridge naik 0,139 dari v5, Palm_Oil_Fruit naik 0,074 — dua kelas prioritas buat deployment ini.
- Person turun 0,334 sebagai konsekuensinya (detail di bawah).
- mAP50 keseluruhan bukan patokan utama yang dipakai buat milih model.
- Data: `yolo val` ulang ke 59 gambar protected-59 yang sama, tanggal 12 Agustus 2026 — bukan angka lama `results.csv` (split val tiap versi beda-beda, nggak bisa dibandingin langsung).
- v9 (0,9673) sengaja dikeluarin dari perbandingan — itu hasil bug data training-validasi kecampur, bukan hasil beneran.

### RGB v12 (`models/rgb_v12_best.pt`)

Dites di data protected-59 (data tetap, aman dari kebocoran, bisa dibandingin ke semua versi dari v10 ke atas):

| Kelas                 | v5 (awal) | v12 (dipakai sekarang) |
| --------------------- | --------- | ---------------------- |
| Bridge                | 0,442     | 0,581                  |
| Car                   | 0,860     | 0,934                  |
| Motorcycle            | 0,826     | 0,738                  |
| Palm_Oil_Fruit        | 0,803     | 0,877                  |
| Person                | 0,866     | 0,532                  |
| Truck                 | 0,880     | 0,913                  |
| **mAP50 Keseluruhan** | **0,779** | **0,763**              |

Data validasi yang lebih besar (295 gambar, nggak bisa dibandingin ke versi lama): mAP50 keseluruhan **0,879**.

#### Tabel acuan v12 yang paling akurat (fresh, 13 Agustus 2026, pakai `conf=0.25` eksplisit)

**mAP50 keseluruhan di threshold operasional (conf=0,25): 0,680** — bukan 0,763 kayak tabel di atas.

| Kelas                 | Gambar | Objek  | Precision | Recall    | mAP50     | mAP50-95  |
| --------------------- | ------ | ------ | --------- | --------- | --------- | --------- |
| Bridge                | 15     | 19     | 0,667     | 0,526     | 0,513     | 0,175     |
| Car                   | 17     | 19     | 0,895     | 0,895     | 0,889     | 0,422     |
| Motorcycle            | 5      | 5      | 1,000     | 0,600     | 0,595     | 0,393     |
| Palm_Oil_Fruit        | 6      | 10     | 0,818     | 0,900     | 0,877     | 0,426     |
| Person                | 5      | 6      | 1,000     | 0,333     | 0,335     | 0,050     |
| Truck                 | 16     | 17     | 0,833     | 0,882     | 0,871     | 0,503     |
| **Semua (rata-rata)** | **59** | **76** | **0,869** | **0,689** | **0,680** | **0,328** |

- Tabel di bagian sebelumnya pakai confidence default ultralytics (~0,001), buat dibandingin apel-ke-apel antar versi — tapi ini **nggak** mencerminkan pemakaian beneran (`WORKFLOW.md` bagian 9: confidence default bikin jumlah deteksi keliatan lebih banyak dari kenyataan).
- Tabel ini pakai `conf=0.25` eksplisit — nilai yang beneran dipakai di project ini — jadi ini yang paling representatif buat "v12 sekarang sebagus apa".
- 0,680 vs 0,763 itu dua-duanya benar, cuma ngukur hal yang beda (lihat Limitasi 3 di bawah).
- Dijalanin ulang 13 Agustus 2026, langsung dari `psn_palm_oil_v12/weights/best.pt` ke protected-59.
- Confusion matrix diperbarui: `report/confusion_matrices/rgb_v12_confusion_matrix.png`.

**Batasan yang diterima: Person masih lemah kalau posenya jongkok atau nyembunyiin badan.**

- Ini sudah didokumentasikan, bukan disembunyiin.
- 8 cara berbeda udah dicoba di 2 usaha perbaikan (v14, terus v15/v16) — berhasil bikin Bridge lebih akurat, tapi Person tetap nggak kebenerin tanpa bikin masalah baru.
- Usaha ke-9 (v17: augmentasi 4 gambar training pose jongkok) juga dicoba dan ditolak — berhasil benerin 1 dari 2 kasus yang dipantau, tapi munculin salah deteksi baru di tempat lain.
- **Penyelidikan ini sudah resmi ditutup** — nggak ada rencana coba training baru lagi.
- Satu-satunya jalan keluar beneran: anotasi baru buat pose jongkok/nyembunyi — belum ada datanya.
- Detail: `report/PROGRESS_REPORT.md` bagian 1.5/1.5.1.

### Thermal v1 (`models/thermal_v1_best.pt`)

**Baseline pertama buat kemampuan malam hari — belum pernah dites di rekaman PSN asli.** Ditraining pakai data HIT-UAV (dataset thermal publik); rekaman thermal PSN sendiri belum ada.

| Data | Kelas       | mAP50 |
| ---- | ----------- | ----- |
| Test | Person      | 0,928 |
| Test | Car         | 0,958 |
| Test | Keseluruhan | 0,943 |
| Val  | Person      | 0,917 |
| Val  | Car         | 0,986 |
| Val  | Keseluruhan | 0,952 |

### Model cadangan

`models/rgb_v10_best.pt` — basis fine-tune buat v12, disimpan buat jaga-jaga/rollback kalau perlu.

## 5. Cerita Lengkapnya

Sejarah lengkap ada di `report/PROGRESS_REPORT.md`:

- 2 bug pipeline data yang ditemukan & dibenerin (bug pencocokan frame, bug kebocoran data train/val) — dua-duanya lebih ngaruh ke performa dibanding nambah data atau ngoprek model.
- Eksperimen yang membuktikan anotasi manual lebih bagus dari data luar.
- Penyelidikan Person pose jongkok.
- Hasil baseline thermal.

Bukti visual penyelidikan Bridge/Person: `report/diagnostic_findings/`.

**Audit optimasi tanpa nambah data (13 Agustus 2026, `PROGRESS_REPORT.md` Bagian 5):**

- Setelah 9 percobaan training mentok, dicek apakah bobot v12 yang sekarang bisa dipakai lebih maksimal tanpa training ulang.
- Ketemu 2 hal yang beneran membantu (dites di 236 gambar non-protected, dikonfirmasi aman di pengecekan akhir protected-59):
  - **Threshold confidence beda-beda per kelas** — precision naik lumayan buat Bridge/Motorcycle/Palm_Oil_Fruit/Truck, gratis, nggak nambah biaya komputasi.
  - **Ensemble 4 checkpoint khusus buat Person doang** (v10+v12+v13+v16 voting bareng) — recall Person naik dari 0,775 ke 0,900 di data tuning, kelas lain nggak kepengaruh.
- Kode: `runs/v12_optimized_inference/v12_optimized_inference.py`.

**Divalidasi ulang secara independen, 13 Agustus 2026** (`report/inference_validation/VALIDATION_REPORT.md`):

- Pakai data baru yang beneran bersih dan bisa dicek ulang sendiri (`eval_sets/clean_236/`, 236 gambar / 509 objek, dibuat pakai `scripts/build_clean_val_set.py`).
- Hasil: **disetujui jadi opsi tambahan** — F1 Person +0,065, F1 Bridge +0,034 di data 236 gambar, nggak ada kelas yang F1-nya turun, konsisten juga pas dicek di protected-59.
- Konsekuensi jujur: recall Palm_Oil_Fruit dan Motorcycle turun dikit (ditukar sama precision, F1-nya tetap sama), mAP50 turun sedikit (0,012) — efek samping yang sudah diduga dari naikin batas confidence.
- `models/rgb_v12_best.pt` sendiri terbukti nggak berubah sama sekali (checksum sama persis dari awal sampai akhir).

**Sudah diterapkan sebagai opsi tambahan, 13 Agustus 2026:**

- `auto_label.py` sekarang punya flag `--inference-policy vanilla|optimized`, default-nya `vanilla` — cara pakai lama tetap jalan sama persis.
- Cara pakainya ada di bagian 6 di bawah.
- Mode `optimized` manggil langsung kode yang sudah divalidasi (`runs/v12_optimized_inference/v12_optimized_inference.py`), bukan nulis ulang logikanya.

## 6. Auto-Labeling (`scripts/auto_label.py`) — ada kendala jaringan

`scripts/auto_label.py` jalanin model yang sudah ditraining buat prediksi task-task di Label Studio, hasilnya dikirim balik sebagai **prediction** (lihat `WORKFLOW.md` bagian 10 dan penjelasan lengkap di dalam script itu sendiri).

**Pilihan cara kerja — `--inference-policy vanilla|optimized`, default `vanilla`:**

```bash
# default — sama persis kayak sebelum-sebelumnya, nggak ada yang berubah
python auto_label.py --project 4 --unlabeled-only

# opsional — threshold per kelas + ensemble Person yang sudah divalidasi
python auto_label.py --project 4 --unlabeled-only --inference-policy optimized
```

- **`vanilla`** (default): satu model doang (`--weights`, default checkpoint v12 yang lagi dipakai), satu threshold `--conf` buat semua kelas. Perilaku aslinya, nggak kepengaruh apapun yang disebut di bawah ini.
- **`optimized`**: policy yang sudah divalidasi dari `runs/v12_optimized_inference/v12_optimized_inference.py`.
  - Threshold beda-beda per kelas + ensemble 4 checkpoint (v10+v12+v13+v16) khusus buat Person doang.
  - Tetap pakai bobot v12 yang **nggak berubah** — nggak ada training ulang, nggak ada model baru.
  - Sudah divalidasi di data 236 gambar dan dikonfirmasi lagi di protected-59 (`report/inference_validation/VALIDATION_REPORT.md`): precision/recall Person dan Bridge dua-duanya membaik, nggak ada kelas yang F1-nya turun.
  - Konsekuensi, biar jujur: mAP50 turun dikit, recall Palm_Oil_Fruit/Motorcycle turun dikit juga (F1-nya tetap sama).
  - Flag `--weights` dan `--conf` diabaikan di mode ini (bakal muncul pesan kalau sempat diisi).
  - Semua checkpoint yang dibutuhkan policy ini dicek dulu **sebelum** mulai proses task apapun — kalau ada checkpoint rusak/hilang, langsung berhenti dengan pesan error yang jelas, bukan diam-diam balik ke vanilla atau jalan terus sampai hasilnya nol prediction (perbaikan dari review kode 13 Agustus 2026).

**Soal masa depan — "optimized" itu khusus buat v12, bukan permanen.**

- `--inference-policy optimized` nunjuk ke satu entri di dictionary `OPTIMIZED_POLICIES` punya `auto_label.py`.
- Bobot model, cara pakai model (policy), dan hasil validasinya itu tiga hal terpisah yang punya versi masing-masing — data baru atau model baru **nggak otomatis** ngubah apa yang dikerjain `optimized` sekarang.
- Model baru butuh policy sendiri yang divalidasi sendiri + entri baru di dictionary (misal `optimized_v18`) sebelum bisa dipilih lewat `--inference-policy`.
- Desain `auto_label.py` sekarang udah generik buat nampung ini — nambah versi baru nggak perlu nulis ulang semuanya. Belum ada versi kayak gitu sekarang.
- Detail lengkap: bagian "Inference policy" di dalam file `auto_label.py` sendiri.

**Kendala jaringan yang diketahui:** server GPU L40S (<GPU_SERVER>) nggak bisa langsung nyambung ke port 8085 Label Studio di <LABEL_STUDIO_SERVER> — firewall-nya diam-diam nolak koneksinya (ping ke server-nya jalan normal, tapi port-nya nggak bisa diakses). Bukan bug di script-nya, memang jalur jaringannya belum ada.

**Solusi sementara yang disarankan:** jalanin `auto_label.py` dari laptop Windows lokal (`C:\Users\asus\psn-training`) — udah bisa akses Label Studio dengan normal, GPU RTX 4060-nya cukup buat inferensi satu gambar. Copy `models/rgb_v12_best.pt` (~6MB) ke laptop kalau belum ada.

Kalau nanti mau auto-labeling otomatis/terjadwal dari server GPU: minta tim infra PSN buka akses port 8085 khusus dari <GPU_SERVER> — permintaan infra sekali doang, bukan sesuatu yang bisa dibenerin dari kode ini.

## 7. Batasan yang Perlu Diketahui

Semua yang di bawah ini sudah pernah dibahas di bagian lain project ini (`PROGRESS_REPORT.md`, dan hasil pengecekan ulang di sesi ini) — dikumpulin di sini biar gampang dilihat sekaligus.

| # | Batasan | Tingkat keseriusan |
| --- | --- | --- |
| 1 | [Person susah kalau posenya jongkok/nyembunyi](#limitasi-1-person-susah-dideteksi-kalau-jongkok-atau-sembunyi) | Tinggi — kelemahan yang sudah diketahui |
| 2 | [Bridge sering salah deteksi](#limitasi-2-bridge-sering-salah-deteksi) | Sedang — tapi lebih baik dari yang dikira sebelumnya |
| 3 | [Data validasi protected-59 kecil banget](#limitasi-3-data-validasi-protected-59-kecil-banget) | Struktural — ngaruh ke cara baca semua angka di dokumen ini |
| 4 | [Hasil training bisa beda walau data sama](#limitasi-4-hasil-training-bisa-beda-beda-walau-data-sama) | Struktural |
| 5 | [Motorcycle turun dan belum balik lagi](#limitasi-5-motorcycle-turun-dan-belum-balik-lagi) | Prioritas rendah |
| 6 | [Model thermal belum dites di rekaman PSN asli](#limitasi-6-model-thermal-belum-dites-di-video-psn-asli) | Sedang — belum tau hasilnya sampai dites |
| 7 | [Auto-labeling wajib dicek manusia](#limitasi-7-auto-labeling-wajib-dicek-manusia) | Aturan wajib |
| 8 | [Konfigurasi label di auto_label.py belum dicek](#limitasi-8-konfigurasi-label-auto_labelpy-belum-dicek) | Nunggu tindakan user |
| 9 | [Server GPU kejegal firewall ke Label Studio](#limitasi-9-server-gpu-kejegal-firewall-ke-label-studio) | Kendala infra |

#### Limitasi 1: Person susah dideteksi kalau jongkok atau sembunyi

**Recall Person di conf=0,25: 0,333** (2 dari 6 objek di protected-59). Precision tetap 1,000 — model nggak asal nebak, dia emang nggak "nangkep" pose kayak gini.

- Lemah khusus buat pose jongkok/nunduk/nyembunyi; pose berdiri biasa udah bagus.
- Penyebab: contoh training pose jongkok sedikit dari awal, makin "tenggelam" pas ronde anotasi berikutnya nambahin banyak banget contoh pose berdiri.
- **9 cara berbeda udah dicoba, nggak ada yang berhasil tanpa data baru:**
  - v6/v7/v13 — pendekatan umum (fine-tuning, oversampling)
  - v14/v15/v16 — hard-negative mining / ubah backbone
  - v17 — augmentasi khusus 4 gambar training pose jongkok (usaha paling spesifik) — berhasil benerin 1 dari 2 kasus yang dipantau, tapi munculin salah deteksi baru di tempat lain
- **Penyelidikan ini sudah resmi ditutup** — nggak ada rencana coba training baru lagi. Detail: `report/PROGRESS_REPORT.md` bagian 1.5/1.5.1.
- Satu-satunya jalan keluar beneran: anotasi Person pose non-berdiri yang baru — belum ada datanya.
- **Ini beda cerita:** audit terpisah yang belakangan (`PROGRESS_REPORT.md` Bagian 5) nemuin ensemble khusus Person (bukan training baru, cuma cara pakai model) yang naikin recall dari 0,775 ke 0,900 di 236 gambar data uji. Ini bukan buka lagi penyelidikan yang udah ditutup — jalannya beda sama sekali (pakai checkpoint yang udah ada). Lihat bagian 5 di atas.

#### Limitasi 2: Bridge sering salah deteksi

**Precision Bridge: 0,667 di conf=0,25** — 1 dari 3 prediksi Bridge salah (10 benar dari 15 prediksi, 5 salah deteksi ke background, dicek langsung dari confusion matrix).

- Penyebab: tekstur tipis/pucat/bercabang kebaca Bridge (pelepah sawit jatuh, kelokan sungai, landasan drone).
- **Lebih bagus dari angka ~53% yang pernah disebut sebelumnya** — angka lama itu udah basi, jangan dipakai lagi tanpa dicek ulang.
- Datanya kecil (cuma 19 objek Bridge asli di protected-59) — anggap ini perkiraan yang beneran tapi masih agak goyang (lihat Limitasi 3).
- Threshold Bridge naik ke 0,40 (bukan 0,25) bisa benerin ini lebih jauh tanpa ngorbanin recall — sudah dibuktikan di data 236 gambar (`PROGRESS_REPORT.md` bagian 5.2, Bagian 5 di atas).

#### Limitasi 3: Data validasi protected-59 kecil banget

**Protected-59 kecil banget buat beberapa kelas** — baca semua angka di dokumen ini dengan ini di kepala, bukan cuma buat Person doang:

| Kelas          | Jumlah objek |
| -------------- | ------------ |
| Motorcycle     | 5            |
| Person         | 6            |
| Palm_Oil_Fruit | 10           |
| Truck          | 17           |
| Car            | 19           |
| Bridge         | 19           |

- Person (cuma 6) paling sedikit buat kelas yang paling penting secara operasional — tapi Motorcycle (cuma 5) malah lebih sedikit lagi.
- Satu deteksi yang salah aja bisa bikin recall kelas-kelas ini naik-turun 10-20 poin persentase sendirian.

#### Limitasi 4: Hasil training bisa beda-beda walau data sama

**Variasi run-to-run: sekitar 0,10-0,17 poin mAP50** — walaupun data training-nya persis sama.

- v10 dan v11 ditraining pakai data yang sama banget sebagai pengecekan khusus — mAP50 per kelas tetap beda (Bridge 0,517→0,415, Motorcycle 0,880→0,714).
- Kalau nanti ada versi baru yang bedanya cuma beberapa poin mAP50 doang, itu masih masuk rentang "noise" alami ini — jangan buru-buru dianggap beneran membaik atau memburuk.

#### Limitasi 5: Motorcycle turun dan belum balik lagi

**mAP50 Motorcycle turun setelah v10 (0,880), belum pernah balik ke level itu lagi sampai v16.**

- v11 0,714, v12 0,738, ..., yang paling bagus setelahnya v14/v15 sama-sama di 0,872 — masih di bawah v10.
- Bukan turun terus-menerus rapi — lebih ke turun, naik sebagian, tapi nggak sampai balik penuh.
- Penyebab pasti belum ketauan; nggak diselidiki lebih jauh soalnya Motorcycle bukan prioritas utama dibanding Bridge/Person/Palm_Oil_Fruit.

#### Limitasi 6: Model thermal belum dites di video PSN asli

**`psn_thermal_v1` belum pernah dites pakai rekaman thermal kebun sawit PSN yang asli** — soalnya belum ada rekamannya.

- Ditraining dan dites semuanya pakai data HIT-UAV, dataset publik (dari sekolah/tempat parkir).
- Angka benchmark-nya bagus (mAP50 Test 0,943), tapi itu belum tentu sama bagusnya di kondisi nyata sampai beneran dites.

#### Limitasi 7: Auto-labeling wajib dicek manusia

**Semua hasil prediction dari `auto_label.py` wajib dicek/diedit manusia dulu sebelum jadi anotasi resmi — jangan pernah langsung disubmit otomatis.**

- Ini aturan wajib, karena Limitasi 1 dan 2 di atas — khususnya buat Bridge dan Person, bukan sekadar saran.
- Satu-satunya kasus yang mungkin bisa dicek lebih santai (nggak perlu satu-satu): prediction Car yang confidence-nya tinggi banget — precision/recall Car dua-duanya udah di 0,895.

#### Limitasi 8: Konfigurasi label auto_label.py belum dicek

**`from_name`/`to_name` di format prediction JSON buat Label Studio belum dicek kecocokannya sama config asli.**

- Script-nya sekarang pakai nilai default dari template standar Label Studio (`"label"`/`"image"`).
- Nunggu user ngambil `label_config` yang asli lewat terminal langsung dengan aman (jangan pernah ditempel di chat, sesuai catatan keamanan di bagian awal dokumen ini).

#### Limitasi 9: Server GPU kejegal firewall ke Label Studio

**`auto_label.py` nggak bisa dijalanin langsung dari server GPU L40S ke Label Studio — lihat bagian 6 di atas.**

- Harus dijalanin dari laptop lokal atau dari server Label Studio itu sendiri.

## Yang Belum Bisa Dicek

- Apakah ada draft anotasi lain di luar yang udah dihitung di sini — nggak bisa dipastikan (query database di atas cuma ngitung anotasi yang udah disubmit/selesai, draft yang belum disubmit nggak kehitung).
- `report/PRESENTATION_SCRIPT.md`, yang pernah disebut di sejarah project ini, ternyata nggak ketemu filenya waktu semua file dikumpulin — nggak dibikin ulang di sini soalnya nggak diminta secara khusus, tapi dicatat di sini kalau-kalau memang masih dibutuhkan terpisah.
- ~~Confusion matrix thermal (`report/confusion_matrices/thermal_v1_confusion_matrix*.png`) belum dibuat ulang pakai conf=0,25~~ — **sudah dibenerin 13 Agustus 2026**: sudah dibuat ulang pakai `conf=0.25` eksplisit, split test, ngikutin cara yang sama kayak punya RGB (`WORKFLOW.md` bagian 9). Angka terbaru di threshold ini: mAP50 keseluruhan 0,921, mAP50-95 0,594 (Person 0,901/0,479, Car 0,941/0,710) — sedikit lebih rendah dari angka di tabel ringkasan bagian 4 (mAP50 0,943), alasannya sama kayak Limitasi 3: pakai conf=0,25 eksplisit itu motong bagian "ekor" confidence rendah dari kurvanya.
