import spateo as st
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io, segmentation, color,exposure
from scipy.ndimage import distance_transform_edt


adata=ad.read_h5ad('data/Mouse_brain_Adult.h5ad')
watershed_labels=adata.layers['watershed_labels']
print(type(watershed_labels))
print(f"Shape: {watershed_labels.shape}")

# 获取唯一的标签值（除了背景）
unique_labels = np.unique(watershed_labels)
unique_labels = unique_labels[unique_labels != 0]  # 移除背景标签

# 初始化一个和 watershed_labels 大小相同的空数组来存放结果
probability_map = np.zeros_like(watershed_labels, dtype=float)
# check_map=np.zeros_like(watershed_labels, dtype=float)

for label in unique_labels:
    # 为当前细胞核创建一个二值图像
    binary_cell = (watershed_labels == label)
    # 计算细胞核的边界
    # boundaries = segmentation.find_boundaries(binary_cell, mode='outer')
    boundaries = segmentation.find_boundaries(binary_cell, mode='inner')
    
    # 创建一个只包含当前细胞核的距离变换的掩模
    masked_boundaries = np.where(binary_cell, boundaries, 1)

    # 对当前细胞核进行距离变换
    distance = distance_transform_edt(1-masked_boundaries)
    max_distance = distance.max()
    if max_distance == 0:
        # 对于只有一个像素的区域，设置为某个固定的概率值
        probability_map[watershed_labels == label] = 0.5  # 或者其他你希望的值
    else:
        # 正常的归一化过程
        normalized_distance = distance / max_distance
        probability_map[watershed_labels == label] = normalized_distance[watershed_labels == label]
    # print(f"Label {label} max distance: {distance.max()}")
    # 归一化距离并存储到最终的 probability_map 中
    normalized_distance = distance / distance.max()
    probability_map[watershed_labels == label] = normalized_distance[watershed_labels == label]
    
plt.imshow(probability_map*255, cmap="gray")
plt.savefig('fig/probability_map.png')
# 查看形状
print(f"Shape: {probability_map.shape}")

# 查看最大值和最小值
print(f"Max value: {np.max(probability_map)}")
print(f"Min value: {np.min(probability_map)}")

# 查看平均值
print(f"Mean value: {np.mean(probability_map)}")

# 查看中位数
print(f"Median value: {np.median(probability_map)}")

# 查看标准差
print(f"Standard deviation: {np.std(probability_map)}")

# 查看方差
print(f"Variance: {np.var(probability_map)}")