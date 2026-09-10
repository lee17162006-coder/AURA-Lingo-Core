import torch
import librosa
import eng_to_ipa as ipa
import Levenshtein
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_NAME = "facebook/wav2vec2-base-960h"

print("Đang khởi tạo mô hình Wav2Vec2...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

def get_target_ipa(word: str) -> str:
    """Lấy chuỗi IPA chuẩn của từ từ eng-to-ipa"""
    res = ipa.convert(word)
    return res.replace("*", "").replace("ˈ", "").replace("ˌ", "")

def audio_to_ipa(audio_path: str) -> str:
    """Giải mã file âm thanh (.wav) và resample tự động về 16kHz"""
    speech, sample_rate = librosa.load(audio_path, sr=16000)
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
    
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0].lower()
    
    return get_target_ipa(transcription)

def calculate_ipa_similarity(target_ipa: str, spoken_ipa: str) -> float:
    """Tính độ tương đồng IPA (%) dựa trên Levenshtein Distance"""
    if not target_ipa or not spoken_ipa:
        return 0.0
    distance = Levenshtein.distance(target_ipa, spoken_ipa)
    max_len = max(len(target_ipa), len(spoken_ipa))
    similarity = (1 - distance / max_len) * 100
    return round(similarity, 2)

def diagnose_vietnamese_phonetic_errors(target_ipa: str, spoken_ipa: str) -> list:
    """
    BỘ LUẬT PHÂN LOẠI LỖI NGỮ ÂM NGƯỜI VIỆT (Tích hợp từ tài liệu Thành viên 2)
    """
    errors = []
    if not target_ipa or not spoken_ipa:
        return errors

    # --- 1. NHÓM LỖI ÂM TH/ (/θ/ và /ð/) ---
    if 'θ' in target_ipa and 'θ' not in spoken_ipa:
        if 't' in spoken_ipa:
            errors.append("Đọc /θ/ (th) thành âm tắc /t/ (VD: think -> tink)")
        elif 's' in spoken_ipa:
            errors.append("Đọc /θ/ (th) thành âm xát /s/ (VD: think -> sink)")
        elif 'f' in spoken_ipa:
            errors.append("Đọc /θ/ (th) thành âm môi-răng /f/")
        elif 'd' in spoken_ipa:
            errors.append("Đọc /θ/ (th) thành âm hữu thanh /d/")
        else:
            errors.append("Bỏ hoàn toàn âm /θ/ (th) trong từ")

    if 'ð' in target_ipa and 'ð' not in spoken_ipa:
        if 'd' in spoken_ipa:
            errors.append("Đọc /ð/ (th) thành âm tắc /d/ (VD: this -> dis)")
        elif 'z' in spoken_ipa:
            errors.append("Đọc /ð/ (th) thành âm xát /z/")
        elif 't' in spoken_ipa:
            errors.append("Đọc /ð/ (th) thành âm vô thanh /t/")
        elif 'θ' in spoken_ipa:
            errors.append("Đọc /ð/ (th) thành âm vô thanh /θ/")
        else:
            errors.append("Bỏ hoàn toàn âm /ð/ (th) trong từ")

    # --- 2. NHÓM LỖI PHỤ ÂM RÂNG - MÔI - LƯỠI (/r/, /l/, /ʃ/, /s/, /f/, /v/) ---
    if 'r' in target_ipa and 'r' not in spoken_ipa and 'l' in spoken_ipa:
        errors.append("Đọc âm /r/ thành âm bên lợi /l/")
    if 'l' in target_ipa and 'l' not in spoken_ipa and 'r' in spoken_ipa:
        errors.append("Đọc âm /l/ thành âm /r/")

    if 'ʃ' in target_ipa and 'ʃ' not in spoken_ipa and 's' in spoken_ipa:
        errors.append("Đọc âm xát sau lợi /ʃ/ (sh) thành âm xát lợi /s/")
    if 's' in target_ipa and 's' not in spoken_ipa and 'ʃ' in spoken_ipa:
        errors.append("Đọc âm xát lợi /s/ thành âm xát sau lợi /ʃ/ (sh)")

    if 'f' in target_ipa and 'f' not in spoken_ipa and 'p' in spoken_ipa:
        errors.append("Đọc âm môi-răng /f/ thành âm tắc hai môi /p/")
    if 'v' in target_ipa and 'v' not in spoken_ipa:
        if 'w' in spoken_ipa:
            errors.append("Đọc âm môi-răng /v/ thành âm tròn môi /w/")
        elif 'b' in spoken_ipa:
            errors.append("Đọc âm môi-răng /v/ thành âm tắc hai môi /b/")

    # --- 3. NHÓM LỖI BỎ ÂM CUỐI VÀ VÔ THANH HÓA ÂM CUỐI (Final Consonants) ---
    final_consonants = ['p', 't', 'k', 'b', 'd', 'g', 'f', 'v', 's', 'z', 'ʃ', 'tʃ', 'dʒ']
    target_end = target_ipa[-1] if target_ipa else ""
    spoken_end = spoken_ipa[-1] if spoken_ipa else ""

    if target_end in final_consonants:
        # Check bỏ âm cuối
        if not spoken_ipa.endswith(target_end) and target_end not in spoken_ipa[-2:]:
            errors.append(f"Rụng phụ âm cuối /{target_end}/ (Final Consonant Dropping)")
        # Check vô thanh hóa âm cuối (Devoicing)
        elif target_end == 'z' and spoken_end == 's':
            errors.append("Vô thanh hóa âm cuối: Đọc /z/ cuối từ thành /s/")
        elif target_end == 'd' and spoken_end == 't':
            errors.append("Vô thanh hóa âm cuối: Đọc /d/ cuối từ thành /t/")
        elif target_end == 'g' and spoken_end == 'k':
            errors.append("Vô thanh hóa âm cuối: Đọc /g/ cuối từ thành /k/")
        elif target_end == 'v' and spoken_end == 'f':
            errors.append("Vô thanh hóa âm cuối: Đọc /v/ cuối từ thành /f/")

    # --- 4. NHÓM LỖI NGUYÊN ÂM ĐÔI THÀNH NGUYÊN ÂM ĐƠN ---
    diphthongs = ['eɪ', 'aɪ', 'ɔɪ', 'aʊ', 'əʊ', 'oʊ']
    for d in diphthongs:
        if d in target_ipa and d not in spoken_ipa:
            errors.append(f"Đơn giản hóa nguyên âm đôi /{d}/ thành nguyên âm đơn")

    return errors