"""导入必要的库"""
import try_ccnn                         # 调用其中的函数
import cv2                              # 图像操作
import numpy as np                      # 矩阵操作
import skimage.color as color           # 子模块color用于进行颜色空间变换
import time
import math                      # 用于数学运算
import pywt                      # 用于小波变换
import matplotlib.pyplot as plt  # 用于数据可视化
from scipy import signal         # 尝试signal.welch()
from mpl_toolkits.mplot3d import Axes3D

"""摄像头捕获视频"""
cap = cv2.VideoCapture(r'..\video.avi')               # 参数为0时开启摄像头，自带摄像头分辨率为1280*720(宽*高)
ret, image = cap.read()                 # 读取一帧图像，ret是布尔值；image是每一帧图像，三维数组

fps = cap.get(cv2.CAP_PROP_FPS)                 # 获取帧率
print(fps)                                      # HCI的fps=30

frames = []                             # 定义空列表，将image由三维数组改造为四维，加入帧数信息


while ret:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)         # BGR->RGB，opencv读取图像的格式为BGR
    image = color.rgb2yiq(image).astype(np.float32)   # RGB->YIQ,double数据类型用于小波变换
    image = cv2.bilateralFilter(image, 25, 50, 50)            # 双边滤波,需要将图像转换为32位浮点型

    if ret:
        frames.append(image)
    else:
        break
    ret, image = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):  # waitKey() 函数的功能是不断刷新图像，频率时间为delay，单位为ms，返回值为当前键盘按键值；&为按位与
        print("q pressed")  # ord()函数返回字符对应的ASCII数值或者Unicode数值；113 & 0xFF==113
        break
frames = np.array(frames)  # 将frames转换为numpy数组，否则其仍为列表，无shape属性
print(frames.shape)  # 输出为(470,480,640,3),(帧数，帧高，帧宽，通道数)

"""
每50帧图像导入CCNN模型，进行小波变换，生成一幅图像
"""
af = 0.1
ae = 1.0
Ve = 50
d0 = 1e-8   # 设置参数
x0 = 0
y0 = 0
z0 = 0      # 初值赋为0
m = frames.shape[1]                         # 取帧高
n = frames.shape[2]                         # 取帧宽
sequence = np.zeros(250)
video_real = np.zeros((60, 250))         # 尺度=60，样本数=50
img = np.zeros((m, n))
result_img = np.zeros((m, n))
g = frames[0:50, :, :, 1]
g1 = g[625:875, :, :]


"""将前50帧图像导入CCNN模型，打印出ROI点"""
roi_point = []                      # 定义一个空列表存放ROI点
for i in range(0, m, 1):            # i为0-(m-1)之间的整数，y轴
    for j in range(0, n, 1):        # j为0-(n-1)之间的整数，x轴
        for k in range(0, 50, 1):  # k为0-49的整数，代表50帧图像
            sti = g[k, i, j]  # frames是四维数组，取通道1之后，g退化为三维数组
            x0 = math.exp(-af) * x0 + sti
            y0 = math.exp(-ae) * y0 + Ve * z0
            z0 = 1 / (1 + math.exp(-(x0 - y0)))
            sequence[k] = z0
        wavename = "gaus1"  # 小波函数
        cwtmatr, frequencies = pywt.cwt(sequence, np.arange(1, 61), wavename)  # 连续小波变换模块
        len1, len2 = cwtmatr.shape
        for i1 in range(0, len1, 1):
            for j1 in range(0, len2, 1):
                video_real[i1, j1] = cwtmatr[i1, j1].real
        img[i, j] = np.sum(video_real)  # sum()函数无参时，所有全加

        if img[i, j] > 0:
            result_img[i, j] = 255  # 显示白色像素点
            roi_point.append((i, j))
        else:
            result_img[i, j] = 0    # 显示黑色像素点，应为ROI点
            # roi_point.append((i, j))
roi_point = np.array(roi_point)         # 转化为数组
f = open(r"..\result_welch.txt", 'w', encoding='utf-8')   # 将要输出保存的文件地址，若文件不存在，则会自动创建
print(roi_point)
print(len(roi_point), file=f)
print(roi_point.shape)


"""将提取出的ROI点进行PSD分析，分别求取HR估计值，再取众数作为最终HR的估计值"""
def find_list_mode(list):
    """
    计算列表中的众数
    参数：
        list：列表类型，待分析数据
    返回值：
        修改前：list_mode：列表类型，待分析数据的众数，可记录一个或多个众数
        修改后：mode:列表的众数
    """
    list_set = set(list)                            # 取list的集合，去除重复元素
    frequency_dict = {}                             # 创建一个空字典
    for ide in list_set:                              # 遍历每一个list的元素，得到该元素对应的个数.count()
        frequency_dict[ide] = list.count(ide)           # 字典中键为每一个list的元素，值为其在list中出现的次数
    list_mode = []                                  # 创建一个空列表
    for key, value in frequency_dict.items():       # 遍历字典的键值对
        if value == max(frequency_dict.values()):   # keys()、values()分别返回字典中的键、值
            list_mode.append(key)                   # 若某键值对的值为最大值，则添加至列表末尾
    list_mode = np.array(list_mode)                 # 将列表强制转换为数组，才可应用mean()求平均值
    mode = round(list_mode.mean())                  # 求取众数数组的平均值（因为可能出现多个众数的情况）并将其四舍五入为整数
    return mode


hr_list = []                                        # 创建一个空列表，存放每个点经PSD后计算出的心率值
for i in range(0, len(roi_point), 1):
    ypos = roi_point[i, 0]
    xpos = roi_point[i, 1]
    x = g[0:50, ypos, xpos]
    [Pxx, freqs] = signal.welch(x, fs=25, nperseg = 50)   # #Pxx为幅值，freqs为频率
    Pxx = list(Pxx)                                             # 将数组强制转换为列表，才可应用index()求索引
    p = Pxx.index(max(Pxx))                                     # 获取Pxx最大元素的下标
    hr = 60 * freqs[p]                                          # Pxx最大元素对应的频率*60作为hr估计值
    hr_list.append(hr)
HR = round(find_list_mode(hr_list))                             # 取心率列表中的众数
print("心率为：", HR, "次/分", file=f)


"""根据提取出的ROI点波形中的极大值数目测量心率"""
zonghe = 0
for i in range(0, len(roi_point), 1):
    ypos = roi_point[i, 0]
    xpos = roi_point[i, 1]
    X = np.array(g[0:50, ypos, xpos])
    zonghe += len((try_ccnn.find_max(X)))
bpm = round(zonghe / len(roi_point) * 60/10)  # round()实现四舍五入，第二个参数表示要保留的小数位数，取0返回的仍为保留一位小数的浮点型数据
print("心率为：", bpm, "次/分", file=f)
f.close()                                  # 执行此步后才将内容写入文件

"""保存图片"""

cv2.imwrite(r"../test_welch.png", result_img)         # 该路径若有中文，不会报错，但图像无法储存

