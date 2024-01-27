import matplotlib.pyplot as plt
import numpy as np

# 读取数据文件
file_path = 'dataset/task1_result.txt'  # 替换为您的文件路径
data = np.loadtxt(file_path, delimiter='\t',usecols=[0, 1,2])

# 提取坐标和概率值
x = data[:, 0].astype(int)
y = data[:, 1].astype(int)
probabilities = data[:, 2].astype(float)


# 计算统计数据
mean_prob = np.mean(probabilities)
median_prob = np.median(probabilities)
std_prob = np.std(probabilities)
max_prob = np.max(probabilities)
min_prob = np.min(probabilities)

# 显示统计数据
print(f"Mean Probability: {mean_prob}")
print(f"Median Probability: {median_prob}")
print(f"Standard Deviation: {std_prob}")
print(f"Max Probability: {max_prob}")
print(f"Min Probability: {min_prob}")



# 绘制散点图，使用'viridis'彩色渐变
plt.scatter(y, x, c=probabilities, cmap='viridis')
plt.colorbar()  # 显示颜色条

# 设置图表标题和坐标轴标签
plt.title('Probability Visualization')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# 显示图表
plt.savefig('fig/task1_result.png')


# 绘制概率分布图
plt.figure(figsize=(10, 6))
plt.hist(probabilities, bins=50, color='skyblue', edgecolor='black')
plt.title('Probability Distribution of Cell Nuclei')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig('fig/prob_distribution.png')
