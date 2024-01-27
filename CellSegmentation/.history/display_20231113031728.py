import matplotlib.pyplot as plt
import numpy as np

# 读取数据文件
file_path = 'path_to_your_file.txt'  # 替换为您的文件路径
data = np.loadtxt(file_path, delimiter='\t')

# 提取坐标和概率值
x = data[:, 0]
y = data[:, 1]
probabilities = data[:, 2]

# 绘制散点图，使用'viridis'彩色渐变
plt.scatter(x, y, c=probabilities, cmap='viridis')
plt.colorbar()  # 显示颜色条

# 设置图表标题和坐标轴标签
plt.title('Probability Visualization')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# 显示图表
plt.show()
