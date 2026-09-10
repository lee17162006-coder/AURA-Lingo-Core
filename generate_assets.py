import os
import re
import pandas as pd
from gtts import gTTS

AUDIO_DIR = os.path.join("assets", "audio")
EXCEL_FILE = "500 từ vựng tiếng anh A1.xlsx"

def clean_and_generate_audio():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    
    # 1. Xóa các file dạng số (ví dụ: 1.mp3, 491.mp3)
    for file_name in os.listdir(AUDIO_DIR):
        if re.match(r"^\d+\.mp3$", file_name):
            os.remove(os.path.join(AUDIO_DIR, file_name))
    
    if not os.path.exists(EXCEL_FILE):
        print(f"Không tìm thấy {EXCEL_FILE}")
        return

    # 2. Đọc file Excel và tạo file theo tên từ vựng
    df = pd.read_excel(EXCEL_FILE)
    
    # Tìm cột chứa từ vựng tiếng Anh
    word_col = None
    for col in df.columns:
        if "word" in str(col).lower() or "từ" in str(col).lower() or col == df.columns[1]:
            word_col = col
            break
    if word_col is None:
        word_col = df.columns[0]

    print(f"Đang tạo audio chuẩn theo từ vựng (Cột: {word_col})...")
    
    count = 0
    for idx, row in df.iterrows():
        word = str(row[word_col]).strip().lower()
        # Loại bỏ ký tự đặc biệt trong tên file
        safe_word = re.sub(r'[\\/*?:"<>|]', "", word)
        
        if not safe_word or safe_word.isdigit() or safe_word == "nan":
            continue

        file_path = os.path.join(AUDIO_DIR, f"{safe_word}.mp3")

        if not os.path.exists(file_path):
            try:
                tts = gTTS(text=safe_word, lang='en', slow=False)
                tts.save(file_path)
                count += 1
                print(f"[{count}] Đã tạo: {safe_word}.mp3")
            except Exception as e:
                print(f"Lỗi từ '{safe_word}': {e}")

    print("Cập nhật thư mục audio thành công!")

if __name__ == "__main__":
    clean_and_generate_audio()