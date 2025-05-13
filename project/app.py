import streamlit as st
import os
from datetime import datetime
import pandas as pd
from helper import get_base64, set_background, title_style, result_style, create_data_input, save_prediction
import random

# ----------------- Thiết lập cấu hình -----------------
# Thư mục tài nguyên
IMAGE_DIR = "project/images"
BACKGROUND_IMG = os.path.join(IMAGE_DIR, "main.jpg")
EXCEL_LOG = "project/user_input.xlsx"
UPLOAD_DIR = "project/uploaded"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Dữ liệu nhạc cụ (hiển thị và phân loại)
instrument_images = {
    "Đàn bầu": "catru.jpg",
    "Đàn tranh": "chauvan.jpg",
    "Đàn nhị": "cheo.jpg",
    "Sáo": "hatxam.jpg"
}
st.markdown("""
    <style>
        .instrument-container {
            display: flex;
            align-items: center;
            flex-basis : auto;
            justify-content : center;
        }
        .instrument {
            text-align: center;
            margin: 0 20px;
        }
        .instrument img {
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
            width: 100px;
            height: auto;
            transition: transform 0.3s ease;
        }
        .instrument img:hover {
            transform: scale(1.05);
        }
        .instrument-caption {
            margin-top: 10px;
            font-weight: bold;
            font-size: 26px;
            font-family: 'Segoe UI', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- Giao diện -----------------
# Nền
set_background(BACKGROUND_IMG)

# Tiêu đề
st.markdown(title_style, unsafe_allow_html=True)
st.markdown("<div class='title'>Vietnamese Traditional Instrument Classifier</div>", unsafe_allow_html=True)

# Load ảnh nhạc cụ
images = {
    "Đàn bầu": "catru.jpg",
    "Đàn tranh": "chauvan.jpg",
    "Đàn nhị": "cheo.jpg",
    "Sáo": "hatxam.jpg"
}

# Render ảnh
st.markdown("<div class='instrument-container'>", unsafe_allow_html=True)
for caption, filename in images.items():
    image_path = os.path.join(IMAGE_DIR, filename)
    image_b64 = get_base64(image_path)
    st.markdown(f"""
        <div class='instrument'>
            <img src='data:image/png;base64,{image_b64}' alt='{caption}' />
            <div class='instrument-caption'>{caption}</div>
        </div>
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Upload và phân loại -----------------
st.markdown("### 🎵 Upload file âm thanh (.mp3 hoặc .wav)")
uploaded_file = st.file_uploader("Chọn tệp", type=["mp3", "wav"])

# Giả lập mô hình phân loại
def classify_instrument(audio_path):
    return random.choice(list(instrument_images.keys()))

# Nếu có file upload
if uploaded_file:
    st.audio(uploaded_file, format='audio/mp3')
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{uploaded_file.name}")

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("📁 File đã được tải lên!")

    # Phân loại
    if st.button("🎯 Phân loại"):
        result = classify_instrument(save_path)
        st.markdown(result_style, unsafe_allow_html=True)
        st.markdown(f"<div class='result'>🎼 Kết quả: {result}</div>", unsafe_allow_html=True)

        # Tạo file nếu chưa có
        if not os.path.exists(EXCEL_LOG):
            pd.DataFrame(columns=["Filename", "Predicted Instrument"]).to_excel(EXCEL_LOG, index=False)

        existing_data = pd.read_excel(EXCEL_LOG)
        new_data = create_data_input([uploaded_file.name], [result], existing_data, existing_data.shape[0])
        save_prediction(EXCEL_LOG, existing_data, new_data)

        # Thống kê
        st.markdown("### 📊 Thống kê kết quả phân loại:")
        chart_data = pd.read_excel(EXCEL_LOG)
        stats = chart_data["Predicted Instrument"].value_counts()
        st.bar_chart(stats)
