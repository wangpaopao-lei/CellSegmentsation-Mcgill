import numpy as np
from scipy.ndimage import label, distance_transform_edt

# 可以修改的变量
threshold = 0.5  # 用于确定细胞核的概率阈值
min_cell_size = 50  # 新细胞核的最小像素点数

# 假设 task1_result 和 watershed 都是以 NumPy 数组的形式读入
task1_result = np.loadtxt('task1_result.txt')
watershed = np.loadtxt('watershed.txt')

# 处理 task1_result
task1_binary = task1_result[:, 2] >= threshold
task1_labeled, _ = label(task1_binary.reshape(1200, 1200))

# 初始化最终结果数组
final_labels = np.zeros_like(watershed)

# 对比 task1_labeled 和 watershed 的标签
for label_num in np.unique(task1_labeled):
    if label_num == 0:
        continue  # 跳过背景
    mask = task1_labeled == label_num
    overlap = watershed[mask]
    most_common = np.bincount(overlap[overlap > 0]).argmax()

    if most_common:
        final_labels[mask] = most_common
    else:
        # 处理新细胞核
        if mask.sum() > min_cell_size:
            distance = distance_transform_edt(~mask)
            nearest_label = watershed[np.argmin(distance)]
            final_labels[mask] = nearest_label

# 输出结果
task2_result = np.hstack((task1_result, final_labels.reshape(-1, 1)))
np.savetxt('task2_result.txt', task2_result, fmt='%d')
