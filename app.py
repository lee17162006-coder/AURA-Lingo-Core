import os
import tempfile
import streamlit as st
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Tải biến môi trường (Gemini API Key)
load_dotenv()

# Import Gemini SDK
from google import genai
from google.genai import types

# Import module đánh giá ngữ âm
from phonetic_evaluator import (
    get_target_ipa,
    audio_to_ipa,
    calculate_ipa_similarity,
    diagnose_vietnamese_phonetic_errors
)
from audio_recorder_streamlit import audio_recorder

# Khởi tạo Gemini Client
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

st.set_page_config(
    page_title="AURA-Lingo - Phonetic Evaluator",
    page_icon="🎙️",
    layout="wide"
)

# --- TIÊU ĐỀ ỨNG DỤNG ---
st.title("🎙️ AURA-Lingo: Multi-modal Phonetic Evaluator")
st.caption("Hệ thống nhận diện đồ vật qua Camera và chấm điểm phát âm tiếng Anh chuẩn IPA")

# Tách giao diện làm 2 cột
col_vision, col_audio = st.columns([1, 1])

# --- CỘT 1: NHẬN DIỆN VẬT THỂ QUA CAMERA ---
with col_vision:
    st.header("1. Camera & Gemini Vision")
    camera_image = st.camera_input("Chụp hình đồ vật/thẻ từ vựng")
    
    if camera_image:
        image = Image.open(camera_image)
        st.image(image, caption="Ảnh đã chụp", use_container_width=True)
        
        if st.button("🔍 Nhận diện từ vựng (Gemini)", type="primary"):
            if not client:
                st.error("Chưa cấu hình GEMINI_API_KEY trong file .env!")
            else:
                with st.spinner("Gemini đang phân tích hình ảnh..."):
                    try:
                        # Convert ảnh sang byte
                        import io
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='JPEG')
                        img_bytes = img_byte_arr.getvalue()

                        prompt = "Identify the main object in this image. Return ONLY the English noun for the object, in lowercase, with no extra punctuation or text (e.g., 'cat', 'book', 'apple')."
                        
                        response = client.models.generate_content(
                            model='gemini-3.6-flash',
                            contents=[
                                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                                prompt
                            ]
                        )
                        detected_word = response.text.strip().lower()
                        st.session_state['detected_word'] = detected_word
                        st.success(f"Từ vựng nhận diện được: **{detected_word}**")
                    except Exception as e:
                        st.error(f"Lỗi Gemini Vision: {e}")

# Mặc định chọn từ vựng nếu đã nhận diện hoặc cho phép nhập tay
target_word = st.text_input(
    "Từ mục tiêu cần luyện đọc:",
    value=st.session_state.get('detected_word', 'cats')
).strip().lower()

target_ipa = get_target_ipa(target_word) if target_word else ""

# --- CỘT 2: THU ÂM VÀ ĐÁNH GIÁ PHÁT ÂM ---
with col_audio:
    st.header("2. Ghi âm & Chấm điểm ngữ âm")
    
    if target_word:
        st.info(f"Từ luyện tập: **{target_word}** | Phiên âm chuẩn IPA: **/{target_ipa}/**")
    
    st.write("Nhấn vào biểu tượng Micro bên dưới để bắt đầu ghi âm:")
    audio_bytes = audio_recorder(
        text="Bấm để ghi âm",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="2x",
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        # Lưu audio_bytes thành file .wav tạm thời để đưa vào Wav2Vec2
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            temp_wav.write(audio_bytes)
            temp_wav_path = temp_wav.name

        with st.spinner("Đang phân tích âm thanh bằng Wav2Vec2 & kiểm tra IPA..."):
            try:
                spoken_ipa = audio_to_ipa(temp_wav_path)
                similarity = calculate_ipa_similarity(target_ipa, spoken_ipa)

                st.subheader("📊 Kết quả Phân tích Ngữ âm")
                st.metric(label="Độ chính xác IPA", value=f"{similarity}%")

                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.write(f"**IPA Chuẩn:** /{target_ipa}/")
                with col_res2:
                    st.write(f"**IPA Đọc được:** /{spoken_ipa}/")

                # Chẩn đoán lỗi ngữ âm người Việt theo Bộ luật mới
                errors = diagnose_vietnamese_phonetic_errors(target_ipa, spoken_ipa)

                if errors:
                    st.error("❌ Phát hiện lỗi ngữ âm người Việt:")
                    for err in errors:
                        st.write(f"- {err}")
                else:
                    st.success("✅ Phát âm tốt! Không phát hiện lỗi đặc trưng người Việt.")

            except Exception as e:
                st.error(f"Lỗi khi phân tích âm thanh: {e}")
            finally:
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)