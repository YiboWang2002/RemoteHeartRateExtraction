"""计算视频帧中每个点的平均心率，并将结果保存到 CSV 文件中（PSD的Welch方法，绿色通道强度值）"""
import numpy as np
import cv2
from scipy import signal
import csv


def calculate_average_heart_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    signal_length = len(frames)

    # 获取视频帧的形状
    frame_shape = frames[0].shape

    # 创建 CSV 文件并写入标题行
    csv_file = open(r'..\heart_rate_data.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['X', 'Y', 'Heart Rate (bpm)'])

    for x in range(frame_shape[1]):
        for y in range(frame_shape[0]):
            intensity_values = []
            for frame in frames:
                intensity = frame[y, x, 1]  # 绿色通道强度
                intensity_values.append(intensity)
            intensity_values = np.array(intensity_values)

            # 对该点的时序信号作一个带通滤波
            low_freq = 0.7 / (fps / 2)  # 低频截止频率
            high_freq = 2.5 / (fps / 2)  # 高频截止频率
            b, a = signal.butter(3, [low_freq, high_freq], 'bandpass')  # 3阶巴特沃斯带通滤波器
            filtered_signal = signal.filtfilt(b, a, intensity_values)

            # 使用Welch方法计算PSD
            f, psd = signal.welch(filtered_signal, fs=fps, nperseg=70)

            # 寻找对应于心率的峰值频率
            peak_idx = np.argmax(psd)
            heart_rate = round(60 * f[peak_idx])

            # 写入 CSV 文件
            csv_writer.writerow([x, y, heart_rate])

    csv_file.close()

    print("Heart rate data saved to heart_rate_data.csv")


# 应用函数
video_path = r'../roi_video.avi'
calculate_average_heart_rate(video_path)
