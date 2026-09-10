import torch
import eng_to_ipa as ipa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Sử dụng model Wav2Vec2 thuần Python, không phụ thuộc backend eSpeak C++
MODEL_NAME = "facebook/wav2vec2-base-960h"

print("Đang khởi tạo mô hình Wav2Vec2...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

def get_target_ipa(word: str) -> str:
    """Lấy chuỗi IPA chuẩn của từ từ eng-to-ipa"""
    res = ipa.convert(word)
    return res.replace("*", "").replace("ˈ", "").replace("ˌ", "")

def evaluate_final_consonants(target_word: str, spoken_ipa: str):
    """Đối chiếu IPA thu được với IPA chuẩn và chẩn đoán lỗi rụng âm cuối (/s/, /t/, /z/)"""
    target_ipa = get_target_ipa(target_word)
    
    print(f"\n--- PHÂN TÍCH NGỮ ÂM ---")
    print(f"Từ mục tiêu:  {target_word}")
    print(f"IPA chuẩn:    /{target_ipa}/")
    print(f"IPA đọc được: /{spoken_ipa}/")
    
    critical_ending_sounds = ['s', 't', 'z']
    errors = []
    
    if target_ipa:
        last_char_target = target_ipa[-1]
        if last_char_target in critical_ending_sounds:
            if not spoken_ipa.endswith(last_char_target) and last_char_target not in spoken_ipa[-2:]:
                errors.append(f"Thiếu âm cuối /{last_char_target}/ (Final Consonant Dropping)")

    if errors:
        print("\n❌ Phát hiện lỗi ngữ âm:")
        for err in errors:
            print(f" - {err}")
    else:
        print("\n✅ Phát âm chuẩn hoặc không phát hiện lỗi rụng âm cuối nghiêm trọng!")

if __name__ == "__main__":
    print("Mô hình đã sẵn sàng. Chạy thử nghiệm đối chiếu IPA:")
    # Giả lập 2 trường hợp người học đọc từ "cats" (/kæts/)
    evaluate_final_consonants("cats", "kæt")   # Test đọc thiếu âm /s/
    evaluate_final_consonants("cats", "kæts")  # Test đọc chuẩn