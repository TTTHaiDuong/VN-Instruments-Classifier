import os
from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from config import *



def audio_to_mel_spectrogram(file_path, n_mels=128, hop_length=512, n_fft=2048, fixed_length=128, sr=22050, duration=None, input_shape=(128, 128, 3)):
    """Chuyển file âm thanh thành Mel-spectrogram với kích thước cố định.

    Parameters:
        file_path (string): Đường dẫn đến file âm thanh (WAV, MP3, v.v.).
        n_mels (int): Số lượng Mel bands.
        hop_length (int): Khoảng cách giữa các khung.
        n_fft (int): Kích thước FFT.
        fixed_length (int): Kích thước cố định của trục thời gian (ví dụ 128, để tạo ma trận 128×128).
        sr (int): Tần số lấy mẫu khi số hoá tín hiệu analog (sampling rate).

    Returns:
        mel_spec_db (ndarray): Mel-spectrogram dạng log (decibel) với kích thước `n_mels`×`fixed_length`.
    """
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
        
    # Đệm thêm hoặc cắt bớt thời lượng của file
    if duration:
        target_length = int(duration * sr)
        y = np.pad(y, (0, max(0, target_length - len(y))), mode='constant')[:target_length]

    # Chuyển sang mel-spectrogram, log decibel
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)    
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Cắt hoặc đệm thêm cho mel-spectrogram để có kích thước cố định
    if mel_spec_db.shape[1] > fixed_length:
        mel_spec_db = mel_spec_db[:, :fixed_length]
    else:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, fixed_length - mel_spec_db.shape[1])), mode='constant')
    
    # Thêm chiều cho Mel-spectrogram để phù hợp với đầu vào của CNN
    mel_spec_db = mel_spec_db[..., np.newaxis]
    
    # Chuyển đổi kích thước Mel-spectrogram về kích thước đầu vào của mô hình
    mel_spec_db = tf.image.resize(mel_spec_db, input_shape[:2], method='bilinear').numpy()
    
    # Chuyển đổi về kích thước 3 kênh
    if input_shape[-1] == 3:
        mel_spec_db = np.repeat(mel_spec_db, 3, axis=-1)
    
    # Chuẩn hóa giá trị về khoảng [0, 1]
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-10)
    return mel_spec_db



def display_mel_spectrogram(mel_spec_db, sr=22050, hop_length=512):
    """Hiển thị Mel-spectrogram dưới dạng hình ảnh.
    
    Parameters:
        mel_spec_db (ndarray): Mel-spectrogram dạng log.
        sr (int): Tần số lấy mẫu khi số hoá tín hiệu analog (sampling rate).
        hop_length (int): Khoảng cách giữa các khung.
    """
    # Chuẩn hoá chiều của mel_spec_db
    if len(mel_spec_db.shape) == 3:
        if mel_spec_db.shape[-1] == 1:
            mel_spec_db = mel_spec_db[:, :, 0]  # Loại bỏ kênh đơn thành 2D
        elif mel_spec_db.shape[-1] == 3:
            mel_spec_db = mel_spec_db[:, :, 0]  # Lấy kênh đầu tiên để hiển thị
        else:
            raise ValueError(f"Mong đợi 1 hoặc 3 kênh, nhận được kích thước {mel_spec_db.shape}")
    elif len(mel_spec_db.shape) != 2:
        raise ValueError(f"Mong đợi mảng 2D hoặc 3D với 1/3 kênh, nhận được kích thước {mel_spec_db.shape}")
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.show()
    plt.close()



def save_mel_spectrograms(input_dir, output_dir, n_mels=128, hop_length=512, n_fft=2048, sr=22050, duration=5.0, input_shape=(128, 128, 3)):
    """Chuyển các file âm thanh trong input_dir thành Mel-spectrogram và lưu dưới dạng .png trong output_dir.
    
    Parameters:
        input_dir (str): Đường dẫn đến thư mục chứa file âm thanh.
        output_dir (str): Đường dẫn đến thư mục lưu file .png.
        n_mels (int): Số lượng Mel bands.
        hop_length (int): Khoảng cách giữa các khung.
        n_fft (int): Kích thước cửa sổ FFT.
        sr (int): Tần số lấy mẫu.
        duration (float): Độ dài cố định của đoạn âm thanh (giây).
        input_shape (tuple): Kích thước đầu ra của Mel-spectrogram.
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy danh sách các thư mục con (lớp) hoặc coi toàn bộ input_dir là một lớp
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    if not classes:
        classes = ['']  # Nếu không có thư mục con, coi input_dir là thư mục chứa file
    
    for class_name in classes:
        # Đường dẫn thư mục lớp
        class_dir = os.path.join(input_dir, class_name) if class_name else input_dir
        output_class_dir = os.path.join(output_dir, class_name) if class_name else output_dir
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Duyệt qua các file âm thanh
        for f in os.listdir(class_dir):
            if f.endswith(('.wav', '.mp3')):
                file_path = os.path.join(class_dir, f)
                output_path = os.path.join(output_class_dir, f"{Path(f).stem}.png")
                
                try:
                    # Tạo Mel-spectrogram
                    mel_spec = audio_to_mel_spectrogram(
                        file_path,
                        n_mels=n_mels,
                        hop_length=hop_length,
                        n_fft=n_fft,
                        sr=sr,
                        duration=duration,
                        input_shape=input_shape
                    )
                    
                    # Lưu thành file .png
                    plt.imsave(output_path, mel_spec[:, :, 0], cmap='magma')
                    print(f"Saved Mel-spectrogram for {file_path} to {output_path}")
                
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")



def print_prediction_results(predictions, class_names, top_k=4):
    """
    In tên nhạc cụ được dự đoán và xác suất từng lớp.

    Parameters:
        predictions (np.ndarray): Kết quả trả về từ model.predict(), shape (1, num_classes)
        class_names (list): Danh sách tên các lớp, ví dụ: ['dantranh', 'danbau', 'dannhi', 'sao']
        top_k (int): Số lớp muốn hiển thị (mặc định: tất cả)
    """
    # Lấy vector xác suất đầu ra
    probs = predictions[0]  # (num_classes,)
    
    # Lấy chỉ số lớp có xác suất cao nhất
    predicted_index = int(np.argmax(probs))
    predicted_label = class_names[predicted_index]
    predicted_prob = probs[predicted_index]

    print(f"Nhạc cụ được dự đoán: **{predicted_label}** (xác suất: {predicted_prob:.2%})\n")
    print("Xác suất từng lớp:")

    # Sắp xếp theo xác suất giảm dần
    sorted_indices = np.argsort(probs)[::-1]

    for i in sorted_indices[:top_k]:
        label = class_names[i]
        prob = probs[i]
        print(f"  - {label:<10}: {prob:.2%}")



def plot_prediction_probabilities(predictions, class_names):
    """
    Vẽ biểu đồ xác suất dự đoán cho từng lớp.

    Parameters:
        predictions (np.ndarray): Kết quả từ model.predict(), shape (1, num_classes)
        class_names (list): Danh sách tên các lớp, ví dụ: ['dantranh', 'danbau', 'dannhi', 'sao']
    """
    probs = predictions[0]

    # Sắp xếp theo xác suất giảm dần
    sorted_indices = np.argsort(probs)[::-1]
    sorted_labels = [class_names[i] for i in sorted_indices]
    sorted_probs = probs[sorted_indices]

    # Vẽ biểu đồ
    plt.figure(figsize=(8, 4))
    bars = plt.barh(sorted_labels, sorted_probs, color='skyblue')
    plt.xlabel('Xác suất')
    plt.title('Biểu đồ xác suất dự đoán nhạc cụ')
    plt.xlim(0, 1)

    # Hiển thị giá trị phần trăm trên từng cột
    for bar, prob in zip(bars, sorted_probs):
        plt.text(prob + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{prob:.2%}", va='center')

    plt.gca().invert_yaxis()  # Lớp có xác suất cao nhất ở trên
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Hiển thị mel-spectrogram từ file âm thanh
    # audio_file = #file âm thanh
    # mel_spectrogram = audio_to_mel_spectrogram(audio_file, n_mels=128, hop_length=512, n_fft=2048, fixed_length=128)
    # print(f"Mel-spectrogram shape: {mel_spectrogram.shape}")
    # display_mel_spectrogram(mel_spectrogram)

    # Lưu các mel-spectrogram từ trong thư mục âm thanh
    save_mel_spectrograms(
        r"C:\Users\tranh\MyProjects\VN-Instruments-Classifier\rawdata\cut",
        r"C:\Users\tranh\MyProjects\VN-Instruments-Classifier\dataset\train\danbau")
    