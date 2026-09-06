import os
from gtts import gTTS

# Thư mục lưu trữ audio
OUTPUT_DIR = "assets/audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Danh sách từ vựng nền tảng A0/A1 (Mẫu kiểm thử)
VOCAB_LIST = [
    "apple", "banana", "cat", "dog", "elephant", "fish", "girl", "house", "ice", "juice",
    "kite", "lemon", "monkey", "notebook", "orange", "pen", "queen", "robot", "sun", "tree",
    "umbrella", "van", "water", "x-ray", "yellow", "zebra", "book", "car", "door", "eye"
]

def generate_audio():
    print(f"Đang tiến hành pre-render {len(VOCAB_LIST)} file audio từ vựng...")
    for word in VOCAB_LIST:
        file_path = os.path.join(OUTPUT_DIR, f"{word}.mp3")
        if not os.path.exists(file_path):
            tts = gTTS(text=word, lang='en', slow=False)
            tts.save(file_path)
            print(f"-> Đã lưu: {file_path}")
        else:
            print(f"-> Đã tồn tại: {file_path}")
    print("\nHoàn tất khởi tạo dữ liệu âm thanh!")

if __name__ == "__main__":
    generate_audio()