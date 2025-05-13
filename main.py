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

# Hàm tăng cường dữ liệu âm thanh
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5)
])





# Hàm trích xuất Mel-spectrogram
def extract_mel_spectrogram(file_path, n_mels=128, hop_length=512, n_fft=2048, fixed_length=128, augment_data=False):
    y, sr = librosa.load(file_path, sr=22050)
    if augment_data:
        y = augment(y, sample_rate=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if mel_spec_db.shape[1] > fixed_length:
        mel_spec_db = mel_spec_db[:, :fixed_length]
    else:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, fixed_length - mel_spec_db.shape[1])), mode='constant')
    return mel_spec_db





# Hàm tải dữ liệu
def load_data(data_dir, n_mels=128, fixed_length=128, augment_data=False):
    X, y = [], []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for file in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file)
            mel_spec = extract_mel_spectrogram(file_path, n_mels, fixed_length=fixed_length, augment_data=augment_data)
            X.append(mel_spec)
            y.append(label)
    return np.array(X), np.array(y)



def process_mel_spectrogram_image(image_path, target_size=(128, 128)):
    """
    Đọc và xử lý hình ảnh Mel-spectrogram thành ma trận phù hợp với mô hình.
    
    Parameters:
        image_path: Đường dẫn tới file hình ảnh.
        target_size: Kích thước đầu vào của mô hình (mặc định: 128x128).
    
    Returns:
        mel_spec: Ma trận Mel-spectrogram đã xử lý (128, 128, 1).
    """
    # Đọc hình ảnh
    img = Image.open(image_path).convert('L')  # Chuyển sang grayscale
    img = img.resize(target_size, Image.LANCZOS)  # Thay đổi kích thước
    
    # Chuyển thành ma trận NumPy
    mel_spec = np.array(img, dtype=np.float32)
    
    # Đảo ngược giá trị (nếu hình ảnh dùng màu sáng cho giá trị thấp)
    # Trong Mel-spectrogram, giá trị thấp (đen) thường là -80 dB, giá trị cao (trắng) là 0 dB
    mel_spec = np.max(mel_spec) - mel_spec
    
    # Chuẩn hóa về khoảng giá trị tương tự Mel-spectrogram (giả sử -80 đến 0 dB)
    mel_spec = (mel_spec - np.min(mel_spec)) / (np.max(mel_spec) - np.min(mel_spec))  # Chuẩn hóa về [0, 1]
    mel_spec = mel_spec * 80 - 80  # Giả lập thang decibel (-80 đến 0)
    
    # Thêm chiều kênh
    mel_spec = mel_spec[..., np.newaxis]  # (128, 128, 1)
    
    # Chuẩn hóa tương tự dữ liệu huấn luyện
    mel_spec = (mel_spec - np.mean(mel_spec)) / np.std(mel_spec)
    
    return mel_spec





def predict_from_mel_image(image_path, model, label_encoder, n_mels=128, fixed_length=128):
    """Dự đoán nhạc cụ từ hình ảnh Mel-spectrogram.
    
    Parameters:
        image_path: Đường dẫn file hình ảnh Mel-spectrogram.
        model: Mô hình CNN đã huấn luyện.
        label_encoder: Bộ mã hóa nhãn.
        n_mels, fixed_length: Tham số Mel-spectrogram (để đảm bảo kích thước).
    
    Returns:
        predicted_instrument: Nhạc cụ dự đoán.
        probabilities: Xác suất cho từng lớp.
    """
    # Xử lý hình ảnh
    mel_spec = process_mel_spectrogram_image(image_path, target_size=(n_mels, fixed_length))
    
    # Thêm batch dimension
    mel_spec = np.expand_dims(mel_spec, axis=0)  # (1, 128, 128, 1)
    
    # Dự đoán
    predictions = model.predict(mel_spec)[0]
    predicted_label_idx = np.argmax(predictions)
    predicted_instrument = label_encoder.inverse_transform([predicted_label_idx])[0]
    
    # Tạo danh sách xác suất
    probabilities = {label_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(predictions)}
    
    return predicted_instrument, probabilities





# # # Tải và tiền xử lý dữ liệu
# data_dir = 'dataset/train'  # Thay bằng đường dẫn thư mục dữ liệu
# X, y = load_data(data_dir, augment_data=True)





# Mã hóa nhãn
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
# y_categorical = to_categorical(y_encoded)





# # Chuẩn hóa dữ liệu
# X = X[..., np.newaxis]  # Thêm chiều kênh (128, 128, 1)
# X = (X - np.mean(X)) / np.std(X)





# Chia dữ liệu
# X_train, X_temp, y_train, y_temp = train_test_split(X, y_categorical, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)





# Xây dựng mô hình CNN
# # model = models.Sequential([
#     layers.Conv2D(32, (3, 3), padding='same', input_shape=(128, 128, 1)),
#     layers.BatchNormalization(),
#     layers.Activation('relu'),
#     layers.MaxPooling2D((2, 2)),

#     layers.Conv2D(64, (3, 3), padding='same'),
#     layers.BatchNormalization(),
#     layers.Activation('relu'),
#     layers.MaxPooling2D((2, 2)),

#     layers.Conv2D(128, (3, 3), padding='same'),
#     layers.BatchNormalization(),
#     layers.Activation('relu'),
#     layers.MaxPooling2D((2, 2)),

#     layers.Conv2D(256, (3, 3), padding='same'),
#     layers.BatchNormalization(),
#     layers.Activation('relu'),
#     layers.GlobalAveragePooling2D(),

#     layers.Dense(128),
#     layers.BatchNormalization(),
#     layers.Activation('relu'),
#     layers.Dropout(0.5),

#     layers.Dense(len(label_encoder.classes_), activation='softmax')
# ])





# # Biên dịch mô hình
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Callbacks
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)





# Huấn luyện mô hình
# history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
#                     callbacks=[reduce_lr, early_stopping])





# Đánh giá mô hình
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"Test accuracy: {test_accuracy:.4f}")





# Lưu mô hình
# model.save('music_instrument_classifier.h5')





# Hàm dự đoán và so sánh nhạc cụ
def predict_instrument(model, input_file, class_names, is_audio=True):
    """
    Dự đoán loại nhạc cụ từ file âm thanh hoặc Mel-spectrogram.
    
    Args:
        model: Mô hình CNN đã huấn luyện.
        input_file (str): Đường dẫn đến file .wav/.mp3 hoặc .png.
        class_names (list): Danh sách tên lớp (ví dụ: ['dantranh', 'guitar']).
        is_audio (bool): True nếu đầu vào là file âm thanh, False nếu là file .png.
    
    Returns:
        str: Tên lớp dự đoán.
    """
    # Chuẩn bị dữ liệu
    if is_audio:
        mel_spec = audio_to_mel_spectrogram(input_file)
    else:
        mel_spec = audio_to_mel_spectrogram(input_file)
    
    # Thêm chiều batch
    mel_spec = np.expand_dims(mel_spec, axis=0)
    
    # Dự đoán
    predictions = model.predict(mel_spec)
    class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[class_idx]
    
    return predicted_class, predictions[0]



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





# Ví dụ dự đoán và so sánh
if __name__ == "__main__":
    # sample_file = "C:\\Users\\tranh\\Downloads\\BaChuaThac-ChauVan-ThanhNgoan-CHAUVAN.wav"
    # predicted_instrument, probabilities = predict_and_compare_instrument(sample_file, model, label_encoder)
    
    # print(f"Predicted instrument: {predicted_instrument}")
    # print("Probabilities:")
    # for instrument, prob in probabilities.items():
    #     print(f"{instrument}: {prob:.4f}")
    
    # # Hiển thị biểu đồ xác suất
    # plot_probabilities(probabilities)

    model1, _, _ = get_model1(n_class=4)
    model1.load_weights(r"checkpoint\model1\model1_01_0.6582.weights.h5")
    model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


    sample_file = r"C:\Users\tranh\Downloads\Senbonzakura Đàn Tranh (guzheng).mp3"
    predicted_instrument, probabilities = predict_instrument(model1, sample_file, ["dantranh", "danbau", "dannhi", "sao"], is_audio=True)
    
    print(f"Dự đoán nhạc cụ: {predicted_instrument}")
    print(f"Tỉ lệ: {probabilities}")
    # for instrument, prob in probabilities.items():
    #     print(f"{instrument}: {prob:.4f}")