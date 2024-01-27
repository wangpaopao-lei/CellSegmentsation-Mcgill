import numpy as np
import matplotlib.pyplot as plt

# 加载细胞核概率值
task1_result = np.loadtxt('dataset/task1_result.txt')
probabilities = task1_result[:, 2]

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

# 绘制概率分布图
plt.figure(figsize=(10, 6))
plt.hist(probabilities, bins=50, color='skyblue', edgecolor='black')
plt.title('Probability Distribution of Cell Nuclei')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()
