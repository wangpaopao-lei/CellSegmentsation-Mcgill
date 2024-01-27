import matplotlib.pyplot as plt
import numpy as np

# 读取数据文件
file_path = 'dataset/task1_result.txt'  # 替换为您的文件路径
data = np.loadtxt(file_path, delimiter='\t',usecols=[0, 1,2])

# 提取坐标和概率值
x = data[:, 0].astype(int)
y = data[:, 1].astype(int)
probabilities = data[:, 2].astype(float)

# 绘制散点图，使用'viridis'彩色渐变
plt.scatter(y, x, c=probabilities, cmap='viridis')
plt.colorbar()  # 显示颜色条

# 设置图表标题和坐标轴标签
plt.title('Probability Visualization')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# 显示图表
plt.savefig('fig/task1_result.png')
