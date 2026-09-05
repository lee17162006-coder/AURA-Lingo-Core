# AURA-Lingo-Core
Core AI Module for AURA-Lingo - Intelligent Bilingual Assistant for Visually Impaired Learners
# AURA-Lingo Core (Auditory Unrestrained Real-time Assistant)

Mã nguồn bộ xử lý trung tâm cho Dự án **AURA-Lingo** - Trợ lý Thính giác Song ngữ Thời gian thực Hỗ trợ Học Ngoại ngữ Hòa nhập cho Người Khiếm thị.

## 🏗 System Architecture (Kiến trúc Hệ thống)
- **Open-Vocabulary Vision:** YOLO-World / CLIP (Vocabulary = 500 A0 Core Words)
- **Phoneme-Level Speech Assessment:** wav2vec2 CTC + CMUdict IPA Matcher
- **Latency Optimization:** Local Audio Pre-rendering (<200ms for core vocab)
- **Framework:** Python 3.10+, Dify.ai Integration

## 📂 Project Structure
```text
├── assets/
│   └── audio/           # File âm thanh pre-render (Lưu local)
├── modules/
│   ├── phoneme_eval.py  # Module tự xây: Chấm phát âm IPA wav2vec2
│   └── vision_detector.py
├── .gitignore           # Bảo mật API Key & Dữ liệu
├── main.py              # Luồng xử lý chính (Pipeline)
└── requirements.txt     # Các thư viện phụ thuộc
