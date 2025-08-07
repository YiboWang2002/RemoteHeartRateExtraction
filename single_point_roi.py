"""对单个点是否为ROI点的判别"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.color as color
import math
import pywt
import os
import matplotlib.font_manager as fm

# 设置字体
font_path = r'..\Times New Roman.ttf'  # 替换为合适的字体文件路径
prop = fm.FontProperties(fname=font_path, size=20)  # 修改字体大小为20磅

# 用户输入特定点的坐标
x = 200  # 输入 x 坐标
y = 200  # 输入 y 坐标

# 摄像头捕获视频
video_folder = r'..\xx_folder'
video_path = os.path.join(video_folder, 'video.avi')
cap = cv2.VideoCapture(video_path)
ret, image = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)

frames = []
mask = np.zeros(image.shape, np.uint8)
frames_RGB = []

while ret:
    frames_RGB.append(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = color.rgb2yiq(image).astype(np.float32)
    if ret:
        frames.append(image)
    else:
        break
    ret, image = cap.read()

frames = np.asarray(frames)
frames_RGB = np.asarray(frames_RGB)
print(frames.shape)

# 选择特定点的 I 通道原始信号
selected_point_signal = frames[:, y, x, 1]

# 以下是CCNN处理代码，处理 sequence 和进行小波变换
af = 0.1
ae = 1.0
Ve = 50
d0 = 1e-8   # 设置参数
x0 = 0
y0 = 0
z0 = 0      # 初值赋为0
m = frames.shape[1]                         # 取帧高
n = frames.shape[2]                         # 取帧宽
sequence = np.zeros(3)                  # 用几帧图像，序列长度就是几帧
video_real = np.zeros((3, 3))         # 尺度=3，样本数=3
img = np.zeros((m, n))
result_img = np.zeros((m, n))

# 设置颜色条刻度的字体
cbar_font = fm.FontProperties(fname=font_path, size=24)

# c = 0                               # 用于计数经由CCNN生成的图像
g = frames[0:3, :, :, 1]          # 取I通道
for i in range(0, m, 1):            # i为0-(m-1)之间的整数，y轴
    for j in range(0, n, 1):        # j为0-(n-1)之间的整数，x轴
        for k in range(0, 3, 1):  # k为0-2的整数，代表3帧图像
            sti = g[k, i, j]        # frames是四维数组，取通道1之后，g退化为三维数组
            x0 = math.exp(-af) * x0 + sti
            y0 = math.exp(-ae) * y0 + Ve * z0
            z0 = 1 / (1 + math.exp(-(x0 - y0)))
            sequence[k] = z0
        wavename = "gaus1"          # 小波函数，未做说明的话均指“gaus1”
        cwtmatr, frequencies = pywt.cwt(sequence, np.arange(1, 4), wavename)  # 一维连续小波变换模块
        len1, len2 = cwtmatr.shape
        for i1 in range(0, len1, 1):
            for j1 in range(0, len2, 1):
                video_real[i1, j1] = cwtmatr[i1, j1].real
        img[i, j] = np.sum(video_real)  # sum()函数无参时，所有全加
        if img[i, j] > 0:
            result_img[i, j] = 255    # 显示白色像素点
        else:
            result_img[i, j] = 0    # 显示黑色像素点
            # frames_RGB[r, i, j, :] = 0

# 连续小波变换的尺度范围
scales = np.arange(1, 4)

# 绘制 I 通道原始信号图
plt.figure(figsize=(12, 6))
plt.title("Original signal of I channel", fontproperties=prop)
plt.plot(selected_point_signal)
plt.xlabel("Frame", fontproperties=prop)
plt.ylabel("Magnitude", fontproperties=prop)
plt.xticks(fontproperties=prop)
plt.yticks(fontproperties=prop)
# plt.savefig("roi_point_1.png", bbox_inches='tight', dpi=300)

# 绘制 CCNN 处理后的 sequence
plt.figure()
markerline, stemlines, baseline = plt.stem(
    range(len(sequence)), sequence, basefmt=" ", linefmt="blue", markerfmt="bo", label="Sequence Value"
)
plt.setp(stemlines, linewidth=2)
plt.setp(markerline, markersize=8)
plt.xlabel("Frame", fontproperties=prop)
plt.ylabel("Sequence value", fontproperties=prop)

# 设置 x 轴刻度为 1, 2, 3
plt.xticks(ticks=[0, 1, 2], labels=['1', '2', '3'], fontproperties=prop)
plt.yticks(fontproperties=prop)
plt.grid()
plt.tight_layout()
plt.savefig("roi_point_2.png", bbox_inches='tight', dpi=300)

# 绘制小波变换结果的 magnitude scalogram
plt.figure()
im = plt.imshow(np.abs(cwtmatr), extent=[1, len(sequence), scales.max(), scales.min()],
                aspect="auto", cmap="jet")

# 创建颜色条并设置标签和字体
cbar = plt.colorbar(label='Magnitude')
cbar.set_label(label='Magnitude', fontproperties=prop)  # 设置颜色条标签字体

# 设置颜色条刻度字体大小和字体
for t in cbar.ax.get_yticklabels():
    t.set_fontproperties(prop)

plt.xlabel("Frame", fontproperties=prop)
plt.ylabel("Scale", fontproperties=prop)

# 设置 x 轴刻度为 1, 2, 3
plt.xticks(ticks=[1, 2, 3], labels=['1', '2', '3'], fontproperties=prop)
plt.yticks(fontproperties=prop)
plt.grid()
plt.tight_layout()

# 保存图像
plt.savefig("roi_point_3.png", bbox_inches='tight', dpi=300)

# 绘制小波变换结果的实部分布图
plt.figure()
plt.imshow(video_real, extent=[1, len(sequence), scales.max(), scales.min()], aspect="auto", cmap="jet")
plt.colorbar(label="Magnitude")
plt.xlabel("Frame", fontproperties=prop)
plt.ylabel("Scale", fontproperties=prop)

# 设置 x 轴刻度为 1, 2, 3
plt.xticks(ticks=[1, 2, 3], labels=['1', '2', '3'], fontproperties=prop)
plt.yticks(fontproperties=prop)
plt.grid()
plt.tight_layout()
# plt.savefig("roi_point_4.png", bbox_inches='tight', dpi=300)

plt.show()
