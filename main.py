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


# Hàm dự đoán và so sánh nhạc cụ
def predict_instrument(model, input_file, class_names, is_audio=True):
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

    sample_file = r"C:\Users\tranh\Downloads\Senbonzakura Đàn Tranh (guzheng).mp3"
    predicted_instrument = predict_instrument(model, sample_file, ["dantranh", "danbau", "dannhi", "sao"], is_audio=True)

    print_prediction_results(predicted_instrument, CLASS_NAMES, top_k=4)
    plot_prediction_probabilities(predicted_instrument, CLASS_NAMES) 
    # print(f"Dự đoán nhạc cụ: {predicted_instrument}")
    # print(f"Tỉ lệ: {probabilities}")
    # for instrument, prob in probabilities.items():
    #     print(f"{instrument}: {prob:.4f}")