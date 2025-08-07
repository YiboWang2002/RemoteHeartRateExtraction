"""利用.csv文件生成一个热力图"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 读取 CSV 文件
csv_file = open(r'..\heart_rate_data.csv', 'r')
csv_reader = csv.reader(csv_file)
next(csv_reader)  # 跳过标题行

# 提取心率值和坐标
heart_rates = []
x_coordinates = []
y_coordinates = []
for row in csv_reader:
    heart_rates.append(int(row[2]))
    x_coordinates.append(int(row[0]))
    y_coordinates.append(int(row[1]))
csv_file.close()

# 创建热力图的数据数组
x_range = max(x_coordinates) - min(x_coordinates) + 1
y_range = max(y_coordinates) - min(y_coordinates) + 1
heatmap_data = np.zeros((y_range, x_range))

# 将心率值映射到热力图数据数组
for x, y, hr in zip(x_coordinates, y_coordinates, heart_rates):
    # 计算心率值减去63bpm的绝对值，再除以63bpm的结果
    normalized_hr = abs(hr - 64) / 64
    heatmap_data[y, x] = normalized_hr

# 绘制热力图
plt.imshow(heatmap_data, cmap='coolwarm')
plt.colorbar(label='Normalized Heart Rate')
plt.title('Normalized Heart Rate Heatmap')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

plt.show()
