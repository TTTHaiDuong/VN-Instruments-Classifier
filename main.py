import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from PIL import Image
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
import matplotlib.pyplot as plt
from utils import audio_to_mel_spectrogram
from build.model1 import get_model1
from config import *
from utils import *
from collections import Counter
from pydub import AudioSegment



# Hàm dự đoán và so sánh nhạc cụ
def predict_instrument(model, input_file):
    mel_spec = audio_to_mel_spectrogram(
        input_file, 
        n_mels=N_MELS, 
        hop_length=HOP_LENGTH, 
        n_fft=N_FFT, 
        fixed_length=128, 
        sr=SR, 
        duration=None, 
        input_shape=INPUT_SHAPE)
    
    print(mel_spec.shape)
    mel_spec = np.expand_dims(mel_spec, axis=0)
    return model.predict(mel_spec)



# Hàm hiển thị xác suất
def plot_probabilities(probabilities):
    plt.figure(figsize=(10, 4))
    plt.bar(probabilities.keys(), probabilities.values())
    plt.title('Probability of Each Instrument')
    plt.xlabel('Instrument')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def predict(file_path):
    model = tf.keras.models.load_model(r"bestmodel\model1.h5")
    predictions = predict_instrument(model, file_path, ["dantranh", "danbau", "dannhi", "sao"], is_audio=True)

    predicted_index = np.argmax(predictions)  # Lấy chỉ mục của xác suất cao nhất

    # Danh sách nhãn của bạn
    labels = ["Đàn bầu", "Đàn nhị", "Đàn tranh", "Sáo"]

    return labels[predicted_index]



# def audio_to_mel_spectrogram(audio_segment, n_mels=128, hop_length=512, n_fft=2048, fixed_length=128, sr=22050, duration=None, input_shape=(128, 128, 3)):
#     """Chuyển file âm thanh thành Mel-spectrogram với kích thước cố định.

#     Parameters:
#         file_path (str): Đường dẫn đến file âm thanh (WAV, MP3, v.v.).
#         n_mels (int): Số lượng Mel bands.
#         hop_length (int): Khoảng cách giữa các khung.
#         n_fft (int): Kích thước FFT.
#         fixed_length (int): Kích thước cố định của trục thời gian (ví dụ 128, để tạo ma trận 128×128).
#         sr (int): Tần số lấy mẫu khi số hoá tín hiệu analog (sampling rate).

#     Returns:
#         mel_spec_db (ndarray): Mel-spectrogram dạng log (decibel) với kích thước `n_mels`×`fixed_length`.
#     """
#     samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
#     samples = samples / np.iinfo(audio_segment.array_type).max  # Chuẩn hoá thành [-1, 1]

#         # Convert to mono nếu stereo
#     if audio_segment.channels == 2:
#         samples = samples.reshape((-1, 2))
#         samples = samples.mean(axis=1)

#     # Chuyển sang mel-spectrogram, log decibel
#     mel_spec = librosa.feature.melspectrogram(y=samples, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)    
#     mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

#     # Cắt hoặc đệm thêm cho mel-spectrogram để có kích thước cố định
#     if mel_spec_db.shape[1] > fixed_length:
#         mel_spec_db = mel_spec_db[:, :fixed_length]
#     else:
#         mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, fixed_length - mel_spec_db.shape[1])), mode='constant')
    
#     # Thêm chiều cho Mel-spectrogram để phù hợp với đầu vào của CNN
#     mel_spec_db = mel_spec_db[..., np.newaxis]
    
#     # Chuyển đổi kích thước Mel-spectrogram về kích thước đầu vào của mô hình
#     mel_spec_db = tf.image.resize(mel_spec_db, input_shape[:2], method='bilinear').numpy()
    
#     # Chuyển đổi về kích thước 3 kênh
#     if input_shape[-1] == 3:
#         mel_spec_db = np.repeat(mel_spec_db, 3, axis=-1)
    
#     # Chuẩn hóa giá trị về khoảng [0, 1]
#     mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-10)
#     return mel_spec_db



def predict_main_instrument(model, file_path, segment_length_ms=5000):
    """
    Dự đoán nhạc cụ chính trong một file âm thanh.

    Parameters:
        file_path (str): Đường dẫn file âm thanh.
        model: Mô hình phân loại nhạc cụ (CNN).
        predict_function: Hàm xử lý dự đoán, nhận input là đoạn audio, trả về tên nhạc cụ.
        segment_length_ms (int): Độ dài mỗi đoạn cắt (ms).

    Returns:
        Tuple(str, dict): Tên nhạc cụ phổ biến nhất và phân phối dự đoán.
    """
    audio = AudioSegment.from_file(file_path)
    duration = len(audio)
    predictions = []

    for start in range(0, duration, segment_length_ms):
        end = min(start + segment_length_ms, duration)
        segment = audio[start:end]

        # Chuyển đoạn audio sang định dạng đầu vào cho model
        mel_spec = audio_to_mel_spectrogram(
            segment, 
            n_mels=N_MELS, 
            hop_length=HOP_LENGTH, 
            n_fft=N_FFT, 
            fixed_length=128, 
            sr=SR, 
            duration=None, 
            input_shape=INPUT_SHAPE)
    
        mel_spec = np.expand_dims(mel_spec, axis=0)
        
        predicted_index = int(np.argmax(model.predict(mel_spec)[0]))
        predicted_label = ["danbau", "dannhi", "dantranh", "sao"][predicted_index]

        predictions.append(predicted_label)

    # Đếm tần suất từng nhạc cụ
    counter = Counter(predictions)
    most_common = counter.most_common(1)[0][0] if counter else None

    return most_common, dict(counter)



if __name__ == "__main__":
    # sample_file = "C:\\Users\\tranh\\Downloads\\BaChuaThac-ChauVan-ThanhNgoan-CHAUVAN.wav"
    # predicted_instrument, probabilities = predict_and_compare_instrument(sample_file, model, label_encoder)
    
    # print(f"Predicted instrument: {predicted_instrument}")
    # print("Probabilities:")
    # for instrument, prob in probabilities.items():
    #     print(f"{instrument}: {prob:.4f}")
    
    # # Hiển thị biểu đồ xác suất
    # plot_probabilities(probabilities)
    model = tf.keras.models.load_model(r"bestmodel\model1.h5")

    sample_file = r"C:\Users\tranh\Downloads\Về quê.mp3"
    # predicted_instrument = predict_main_instrument(model, sample_file)

    print(f"Dự đoán: {predict_instrument(model, sample_file)}")
    # print_prediction_results(predicted_instrument, CLASS_NAMES, top_k=4)
    # plot_prediction_probabilities(predicted_instrument, CLASS_NAMES) 
    # print(f"Dự đoán nhạc cụ: {predicted_instrument}")
    # print(f"Tỉ lệ: {probabilities}")
    # for instrument, prob in probabilities.items():
    #     print(f"{instrument}: {prob:.4f}")