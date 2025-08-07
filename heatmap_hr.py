"""生成心率热力图"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 设置全局字体属性
font_path = r'..\Times New Roman.ttf'  # 替换为合适的字体文件路径
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

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
    heatmap_data[y, x] = hr

# 根据心率值绘制热力图
plt.imshow(heatmap_data, cmap='coolwarm')
plt.colorbar(label='Heart Rate (bpm)')
plt.title('Heart Rate Heatmap')
plt.xlabel('x')
plt.ylabel('y')

# 高亮接近63bpm的点
threshold = 3
near_63 = np.abs(heatmap_data - 69) < threshold
heatmap_data = np.ma.masked_where(~near_63, heatmap_data)
plt.imshow(heatmap_data, cmap='coolwarm', alpha=0.8)

# 调整底部边距
plt.subplots_adjust(bottom=0.15)

plt.savefig('heatmap.png', dpi=600)
plt.show()
