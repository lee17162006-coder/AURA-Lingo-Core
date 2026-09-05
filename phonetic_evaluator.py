import os
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from dotenv import load_dotenv

# Tải biến môi trường từ file .env
load_dotenv()

class PhoneticEvaluator:
    def __init__(self, model_name="facebook/wav2vec2-xlsr-53-espeak-cv-ft"):
        """Khởi tạo mô hình Wav2Vec2 chấm điểm IPA"""
        print("Đang tải mô hình Wav2Vec2...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.eval()
        print("Mô hình Wav2Vec2 đã sẵn sàng!")

    def evaluate_audio(self, audio_path, target_ipa):
        """
        Đọc file âm thanh người dùng nói và so sánh chuỗi IPA với chuỗi chuẩn
        """
        waveform, sample_rate = torchaudio.load(audio_path)

        # Chuyển đổi sample rate về 16kHz nếu cần
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        input_values = self.processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_ipa = self.processor.batch_decode(predicted_ids)[0]

        is_matched = predicted_ipa.strip() == target_ipa.strip()
        
        return {
            "predicted_ipa": predicted_ipa,
            "target_ipa": target_ipa,
            "is_matched": is_matched
        }

if __name__ == "__main__":
    evaluator = PhoneticEvaluator()
    print("Khởi tạo module thành công!")