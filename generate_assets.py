import os
import pandas as pd
from gtts import gTTS

# Tạo thư mục lưu trữ nếu chưa có
AUDIO_DIR = os.path.join("assets", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# Đường dẫn file Excel chứa 500 từ vựng
EXCEL_FILE = "500 từ vựng tiếng anh A1.xlsx"

def generate_audio_assets():
    if not os.path.exists(EXCEL_FILE):
        print(f"Không tìm thấy file {EXCEL_FILE}")
        return

    # Đọc dữ liệu từ Excel
    df = pd.read_excel(EXCEL_FILE)
    
    # Giả định cột đầu tiên chứa từ tiếng Anh (hoặc điều chỉnh tên cột cho đúng)
    word_column = df.columns[0] 
    
    print("Đang khởi tạo bộ 500 file Audio Pre-render...")
    count = 0

    for idx, row in df.iterrows():
        word = str(row[word_column]).strip().lower()
        if not word or pd.isna(word):
            continue

        # Đặt tên file chuẩn hóa theo từ vựng (ví dụ: apple.mp3)
        file_path = os.path.join(AUDIO_DIR, f"{word}.mp3")

        # Skip nếu file đã tồn tại để tiết kiệm thời gian
        if not os.path.exists(file_path):
            try:
                tts = gTTS(text=word, lang='en', slow=False)
                tts.save(file_path)
                count += 1
                print(f"[{count}] Đã tạo: {word}.mp3")
            except Exception as e:
                print(f"⚠️ Lỗi khi tạo audio cho từ '{word}': {e}")

    print(f"\nHoàn tất! Đã khởi tạo thành công bộ file audio trong {AUDIO_DIR}")

if __name__ == "__main__":
    generate_audio_assets()