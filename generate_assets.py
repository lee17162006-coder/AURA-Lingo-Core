import os
import pandas as pd
from gtts import gTTS

OUTPUT_DIR = "assets/audio"
EXCEL_PATH = "500 từ vựng tiếng anh A1.xlsx"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_audio_from_excel():
    if not os.path.exists(EXCEL_PATH):
        print(f"Lỗi: Không tìm thấy file '{EXCEL_PATH}' trong thư mục dự án!")
        return

    # Đọc dữ liệu từ file Excel
    df = pd.read_excel(EXCEL_PATH)
    
    # Tìm cột chứa từ vựng tiếng Anh
    word_column = None
    for col in df.columns:
        if str(col).strip().lower() in ['word', 'words', 'từ vựng', 'tu_vung', 'từ']:
            word_column = col
            break

    if not word_column:
        word_column = df.columns[0] # Lấy cột đầu tiên nếu không tìm thấy đúng tên

    vocab_list = df[word_column].dropna().astype(str).str.strip().tolist()
    print(f"Đã tìm thấy {len(vocab_list)} từ vựng. Bắt đầu pre-render audio...")

    count_new = 0
    for word in vocab_list:
        clean_word = word.lower().strip()
        # Bỏ qua dòng tiêu đề nếu bị lẫn vào
        if clean_word in ['word', 'từ vựng', 'tu_vung']:
            continue
            
        file_path = os.path.join(OUTPUT_DIR, f"{clean_word}.mp3")
        
        if not os.path.exists(file_path):
            try:
                tts = gTTS(text=clean_word, lang='en', slow=False)
                tts.save(file_path)
                count_new += 1
                print(f"-> Đã tạo mới: {file_path}")
            except Exception as e:
                print(f"-> Lỗi khi tạo audio cho từ '{clean_word}': {e}")
        else:
            print(f"-> Đã có sẵn: {file_path}")

    print(f"\nHoàn tất! Đã xử lý toàn bộ danh sách {len(vocab_list)} từ.")

if __name__ == "__main__":
    generate_audio_from_excel()