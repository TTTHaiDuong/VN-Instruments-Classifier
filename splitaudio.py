import os
from pydub import AudioSegment

def split_audio_files(input_dir, output_dir, segment_length=5000):
    """
    Cắt các file âm thanh trong thư mục `input_dir` thành các đoạn nhỏ có độ dài `segment_length` (mặc định 5s),
    và lưu chúng vào thư mục `output_dir`.
    
    Parameters:
        input_dir (str): Thư mục chứa các file âm thanh gốc.
        output_dir (str): Thư mục lưu các file âm thanh đã cắt.
        segment_length (int): Độ dài mỗi đoạn cắt (đơn vị: mili giây, mặc định là 5000ms = 5s).
    """
    # Tạo thư mục đích nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lấy danh sách các file trong thư mục nguồn
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        if not os.path.isfile(file_path):
            continue

        # Đọc file âm thanh
        try:
            audio = AudioSegment.from_file(file_path)
        except Exception as e:
            print(f"Không thể đọc file {filename}: {e}")
            continue
        
        # Kiểm tra nếu độ dài file nhỏ hơn segment_length thì bỏ qua
        if len(audio) < segment_length:
            print(f"File {filename} quá ngắn (< {segment_length / 1000}s), bỏ qua.")
            continue
        
        # Cắt file thành các đoạn nhỏ
        num_segments = len(audio) // segment_length
        
        for i in range(num_segments):
            start_time = i * segment_length
            end_time = start_time + segment_length
            segment = audio[start_time:end_time]
            
            # Tạo tên file mới cho đoạn cắt
            new_filename = f"{os.path.splitext(filename)[0]}_part{i+1}.wav"
            new_file_path = os.path.join(output_dir, new_filename)

            # Lưu đoạn âm thanh cắt ra
            segment.export(new_file_path, format="wav")
            print(f"Đã lưu file: {new_file_path}")



if __name__ == "__main__":
    split_audio_files(
        input_dir=r'C:\Users\tranh\MyProjects\VN-Instruments-Classifier\rawdata\train\danbau',
        output_dir=r'C:\Users\tranh\MyProjects\VN-Instruments-Classifier\rawdata\cut',
        segment_length=5000  # Độ dài mỗi đoạn cắt (5 giây)
    )