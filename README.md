# AutomationPalmOil

**Sistem deteksi objek otomatis dari rekaman drone, buat bantu ngawasin kebun sawit dari udara.**

Bayangin ada drone yang muter-muter kebun sawit tiap hari, ngerekam video. Daripada ada orang yang harus nonton semua videonya satu-satu buat nyari hal mencurigakan, sistem ini otomatis "melihat" videonya dan bilang: "di sini ada orang", "di situ ada mobil", "itu jembatan", dst. Tujuannya bantu deteksi dini aktivitas mencurigakan (contohnya pencurian buah sawit) tanpa perlu nonton rekaman manual.

Ini bukan produk jadi yang tinggal pakai — ini adalah **model AI + kumpulan script otomasi** yang dipakai buat proses melabeli data, melatih model, dan menjalankan deteksinya.

---

## Ada 2 Versi, Buat 2 Kondisi Cahaya Berbeda

| | Siang Hari (RGB) | Malam Hari (Thermal) |
|---|---|---|
| Kamera yang dipakai | Kamera biasa | Kamera panas (thermal/infrared) |
| Jenis objek yang dikenali | 6 jenis: Jembatan, Mobil, Motor, Buah Sawit, Orang, Truk | 2 jenis: Orang, Mobil |
| Sumber data latihan | Rekaman drone asli, dilabeli manual | Dataset publik (belum pernah dites di rekaman kebun sawit asli) |
| Status | Sudah dipakai, versi ke-12 | Baru versi pertama (prototipe) |

Kenapa thermal cuma 2 jenis objek? Karena belum ada dataset thermal publik yang punya contoh jembatan atau buah sawit — dua objek ini terlalu spesifik buat konteks kebun sawit.

---

## Seberapa Bagus Model Ini?

Diukur pakai 2 hal: **Precision** (kalau nebak, seberapa sering itu bener) dan **Recall** (dari semua yang beneran ada, seberapa banyak yang berhasil ketangkep). Nilai 0–1, makin dekat ke 1 makin bagus.

| Jenis Objek | Precision | Recall | Catatan |
|:---|---:|---:|:---|
| Mobil | 0,895 | 0,895 | Bagus, dua-duanya seimbang |
| Buah Sawit | 0,818 | 0,900 | Bagus |
| Truk | 0,833 | 0,882 | Bagus |
| Motor | 1,000 | 0,600 | Kalau nebak nggak pernah salah, tapi 4 dari 10 motor kelewat nggak kedeteksi |
| Jembatan | 0,667 | 0,526 | Masih sering salah tebak (pelepah sawit jatuh/kelokan sungai suka dikira jembatan) |
| Orang | 1,000 | 0,333 | Nggak pernah asal tebak, tapi 2 dari 3 orang yang beneran ada malah kelewat — terutama kalau posenya jongkok/nyembunyi |

**Yang paling penting dari tabel ini: kelas Orang paling lemah justru di kasus yang paling penting** (orang lagi nyembunyi/jongkok, bukan berdiri biasa). Ini batasan yang diketahui dan belum terpecahkan — sudah dicoba 9 cara berbeda buat benerin, belum ada yang berhasil tanpa data anotasi baru.

---

## Cara Kerjanya (Ringkas)

Alur lengkapnya, dari video mentah sampai jadi model yang dipakai, ada di **[WORKFLOW.md](./WORKFLOW.md)**. Garis besarnya:

```
Video drone → potong jadi foto → foto dilabeli (dibantu AI) → dicek manusia
   → dipakai buat melatih model → model diuji → kalau bagus, dipakai
```

---

## Cara Pakai Script Otomasi

Script utama ada di folder [`scripts/`](./scripts):

| Script | Fungsinya |
|---|---|
| `extract_frames.py` | Potong video drone jadi foto-foto diam |
| `label_studio_import.py` | Kirim foto-foto ke sistem pelabelan (Label Studio) |
| `auto_label.py` | Jalankan model AI buat nebak isi foto, hasilnya jadi draft yang dicek manusia |
| `prepare_dataset.py` | Ubah data yang sudah dilabeli jadi format yang bisa dipakai buat melatih model |
| `build_clean_val_set.py` | Bikin set data khusus buat uji coba/tuning, terpisah dari data ujian utama |
| `test_auto_label_policy.py` | Tes otomatis buat mastiin `auto_label.py` masih jalan bener sebelum dipakai |

Contoh pemakaian yang paling sering dipakai — nebak label buat foto-foto yang belum dilabeli:

```bash
export LABEL_STUDIO_API_TOKEN=<token_akses_anda>
python scripts/auto_label.py --project 4 --unlabeled-only
```

**Penting:** hasil dari `auto_label.py` itu cuma **draft/usulan**, bukan jawaban final. Wajib dicek manusia dulu sebelum jadi label resmi — alasannya dijelasin lengkap di [WORKFLOW.md](./WORKFLOW.md#kenapa-nggak-langsung-jadi-anotasi-resmi).

---

## Batasan yang Perlu Diketahui

- **Orang yang jongkok/nyembunyi masih sering kelewat kedeteksi** (lihat tabel di atas). Ini batasan data, bukan batasan cara kerja model — butuh lebih banyak contoh foto pose non-berdiri.
- **Jembatan sering salah tebak** — sekitar 1 dari 3 tebakan itu salah.
- **Data ujian buat sebagian kelas masih sedikit** (Motor dan Orang cuma punya beberapa contoh di data ujian resminya) — jadi satu tebakan salah aja bisa bikin angka performa naik-turun drastis.
- **Model thermal belum pernah dites di rekaman kebun sawit asli** — masih dilatih dan dites pakai data publik dari lokasi lain.
- **Semua hasil deteksi otomatis wajib dicek manusia** sebelum jadi data resmi — ini aturan, bukan saran.

---

## Struktur Repo

```
AutomationPalmOil/
├── README.md          # Dokumen ini
├── WORKFLOW.md         # Penjelasan detail alur kerja + diagram
└── scripts/
    ├── extract_frames.py
    ├── label_studio_import.py
    ├── auto_label.py
    ├── prepare_dataset.py
    ├── build_clean_val_set.py
    └── test_auto_label_policy.py
```
