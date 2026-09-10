import torch
import soundfile as sf
import eng_to_ipa as ipa
import Levenshtein
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Sử dụng mô hình Wav2Vec2 thuần Python (không dính eSpeak dependency)
MODEL_NAME = "facebook/wav2vec2-base-960h"

print("Đang khởi tạo mô hình Wav2Vec2 Speech-to-Text...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

def get_target_ipa(word: str) -> str:
    """Lấy chuỗi IPA chuẩn của từ từ eng-to-ipa"""
    res = ipa.convert(word)
    return res.replace("*", "").replace("ˈ", "").replace("ˌ", "")

def audio_to_ipa(audio_path: str) -> str:
    """Giải mã file âm thanh (.wav) người dùng đọc thành chuỗi IPA dự đoán"""
    # 1. Đọc file âm thanh
    speech, sample_rate = sf.read(audio_path)
    
    # 2. Tiền xử lý dữ liệu âm thanh (chuẩn hóa về 16kHz)
    input_values = processor(speech, sampling_rate=sample_rate, return_tensors="pt").input_values
    
    # 3. Dự đoán ký tự văn bản bằng Wav2Vec2
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0].lower()
    
    # 4. Chuyển đổi văn bản nhận diện được sang IPA
    spoken_ipa = get_target_ipa(transcription)
    return spoken_ipa

def calculate_ipa_similarity(target_ipa: str, spoken_ipa: str) -> float:
    """Tính độ tương đồng IPA (%) dựa trên Levenshtein Distance"""
    if not target_ipa or not spoken_ipa:
        return 0.0
    distance = Levenshtein.distance(target_ipa, spoken_ipa)
    max_len = max(len(target_ipa), len(spoken_ipa))
    similarity = (1 - distance / max_len) * 100
    return round(similarity, 2)

def evaluate_speech_file(target_word: str, audio_path: str):
    """
    Hàm tổng hợp: Đánh giá file âm thanh thực tế với từ mục tiêu
    - Bắt lỗi rụng âm cuối (/s/, /t/, /z/)
    - Tính điểm tương đồng IPA
    """
    target_ipa = get_target_ipa(target_word)
    spoken_ipa = audio_to_ipa(audio_path)
    similarity_score = calculate_ipa_similarity(target_ipa, spoken_ipa)
    
    print(f"\n--- KẾT QUẢ ĐÁNH GIÁ PHÁT ÂM ---")
    print(f"Từ mục tiêu:      {target_word}")
    print(f"IPA mục tiêu:     /{target_ipa}/")
    print(f"IPA nhận diện:    /{spoken_ipa}/")
    print(f"Độ chính xác:     {similarity_score}%")
    
    # Bắt lỗi rụng phụ âm cuối
    critical_ending_sounds = ['s', 't', 'z']
    errors = []
    
    if target_ipa:
        last_char_target = target_ipa[-1]
        if last_char_target in critical_ending_sounds:
            if not spoken_ipa.endswith(last_char_target) and last_char_target not in spoken_ipa[-2:]:
                errors.append(f"Thiếu phụ âm cuối /{last_char_target}/ (Final Consonant Dropping)")

    if errors:
        print("\n❌ Lỗi ngữ âm phát hiện:")
        for err in errors:
            print(f" - {err}")
    else:
        print("\n✅ Phát âm chuẩn hoặc không phát hiện lỗi rụng âm cuối nghiêm trọng.")

if __name__ == "__main__":
    print("Module Phonetic Evaluator (Audio File Input) đã sẵn sàng.")