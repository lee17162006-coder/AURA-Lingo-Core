import os
import glob
import time
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Lỗi: Chưa cấu hình GEMINI_API_KEY trong file .env")
    exit(1)

client = genai.Client(api_key=api_key)

IMAGE_DIR = "dataset_test"
EXCEL_PATH = "500 từ vựng tiếng anh A1.xlsx"

# Đọc danh sách từ A1
df = pd.read_excel(EXCEL_PATH)
word_col = df.columns[0]
for col in df.columns:
    if str(col).strip().lower() in ['word', 'words', 'từ vựng', 'tu_vung', 'từ']:
        word_col = col
        break
vocab_list = [str(w).strip().lower() for w in df[word_col].dropna().tolist()]
vocab_str = ", ".join(vocab_list[:150])

# Lấy danh sách ảnh
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext.upper())))

image_paths = sorted(list(set(image_paths)))
print(f"Tìm thấy {len(image_paths)} ảnh. Bắt đầu benchmark chuẩn hóa từ vựng A1...\n")

prompt = f"""Bạn là trợ lý thị giác AURA-Lingo. 
Xác định đồ vật chính trong ảnh và ƯU TIÊN chọn 1 từ đơn giản nhất thuộc danh sách từ vựng A1 sau:
[{vocab_str}]

Chỉ trả về đúng 1 dòng theo định dạng:
OBJECT: <tên_đồ_vật_tiếng_Anh>
"""

results = []
for idx, img_path in enumerate(image_paths, 1):
    file_name = os.path.basename(img_path)
    
    success = False
    retries = 0
    while not success and retries < 5:
        try:
            with open(img_path, "rb") as f:
                image_bytes = f.read()
                
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt
                ]
            )
            detected = response.text.strip()
            print(f"[{idx}/{len(image_paths)}] {file_name} -> {detected}")
            results.append({"file": file_name, "result": detected})
            success = True
            time.sleep(4.5)  # Giữ tốc độ an toàn (~13 req/min < 20 req/min)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                retries += 1
                print(f"⚠️ Chờ khôi phục Rate Limit (thử lại {retries}/5 trong 30s)...")
                time.sleep(30)
            else:
                print(f"[{idx}/{len(image_paths)}] Lỗi xử lý {file_name}: {e}")
                break

print("\nHoàn tất benchmark Gemini Vision!")