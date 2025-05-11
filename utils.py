import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from config import *

def audio_to_mel_spectrogram(file_path, n_mels=128, hop_length=512, n_fft=2048, fixed_length=128, sr=SR):
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

    # Tải file âm thanh
    y, sr = librosa.load(file_path, sr=sr)
    
    # Tính Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    
    # Chuyển sang log scale (decibel)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Chuẩn hóa kích thước
    if mel_spec_db.shape[1] > fixed_length:
        mel_spec_db = mel_spec_db[:, :fixed_length]  # Cắt nếu quá dài
    else:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, fixed_length - mel_spec_db.shape[1])), mode='constant')  # Đệm nếu quá ngắn
    
    mel_spec_db[..., np.newaxis]
    
    return mel_spec_db



def display_mel_spectrogram(mel_spec_db, sr=SR, hop_length=512):
    """Hiển thị Mel-spectrogram dưới dạng hình ảnh.
    
    Parameters:
        mel_spec_db (ndarray): Mel-spectrogram dạng log.
        sr (int): Tần số lấy mẫu khi số hoá tín hiệu analog (sampling rate).
        hop_length (int): Khoảng cách giữa các khung.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    audio_file = "C:\\Users\\tranh\\Downloads\\BaChuaThac-ChauVan-ThanhNgoan-CHAUVAN.wav"
    
    mel_spectrogram = audio_to_mel_spectrogram(audio_file, n_mels=128, hop_length=512, n_fft=2048, fixed_length=128)
    
    print(f"Mel-spectrogram shape: {mel_spectrogram.shape}")
    
    display_mel_spectrogram(mel_spectrogram)