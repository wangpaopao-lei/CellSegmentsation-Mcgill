import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

with np.load('dataset/expression_data.npz') as data:
    all_exp_merged_bins = data['expression_data']
patchsizey=1200
# 第一步：读取 task2_result 数据
task2_result = np.loadtxt('results/task2_result.txt')


task2_matrix = np.zeros((400, 400))
segmentation_image = np.zeros((400, 400, 3), dtype=np.uint8)
for row in task2_result:
    x, y, label1 = int(row[0]), int(row[1]), int(row[3])
    task2_matrix[x, y] = label1
# 遍历每个像素点，为每个细胞核上的像素点分配不同的颜色
unique_cells = np.unique(task2_result[:, 3])
num_colors = int(max(unique_cells)) + 1  # 转换为整数
colors = plt.cm.jet(np.linspace(0, 1, num_colors))
# 确保颜色数组足够长以覆盖所有唯一的细胞核标签
if len(colors) <= max(unique_cells):
    colors = plt.cm.jet(np.linspace(0, 1, max(unique_cells) + 1))

for cell in unique_cells:
    if cell == 0:  # 跳过背景
        continue
    cell_index = int(cell)
    if cell_index >= len(colors):
        continue
    mask = task2_result[:, 3] == cell
    x, y = task2_result[mask, :2].T
    x, y = x.astype(int), y.astype(int)
    # 逐个像素点设置颜色
    for xi, yi in zip(x, y):
        segmentation_image[yi, xi] = colors[cell_index][:3] * 255
# 显示和保存图像
plt.imshow(task2_matrix)
plt.axis('off')  # 不显示坐标轴
plt.savefig('results/task2_result.png', bbox_inches='tight', pad_inches=0)

cell_sizes = []
for cell in unique_cells:
    if cell == 0:  # 跳过背景
        continue
    mask = task2_result[:, 3] == cell
    cell_size = np.sum(mask)
    cell_sizes.append(cell_size)
plt.figure()
# 绘制细胞大小的分布图
plt.hist(cell_sizes, bins=50, color='blue', alpha=0.7)
plt.title('Cell Size Distribution')
plt.xlabel('Cell Size (number of pixels)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('fig/nucleus_size_distribution.png')


# 第二步：计算每个细胞核的加权质心、总基因表达
unique_cells = np.unique(task2_result[:, 3])
centroids = {cell: [0, 0, 0] for cell in unique_cells if cell != 0}  # 排除背景
nucleus_expression = {cell: np.zeros(all_exp_merged_bins.shape[1]) for cell in centroids if cell != 0}
nucleus_pixel_count = {cell: 0 for cell in centroids if cell != 0}

for cell in unique_cells:
    if cell == 0:
        continue
    mask = task2_result[:, 3] == cell
    x, y, prob = task2_result[mask, :3].T
    total_prob = np.sum(prob)
    centroids[cell] = [np.sum(x * prob) / total_prob, np.sum(y * prob) / total_prob, total_prob]
    
    # 计算细胞核的总基因表达向量
    for xi, yi in zip(x, y):
        idx = int(math.floor(xi) * math.ceil(patchsizey) + math.floor(yi))
        nucleus_expression[cell] += all_exp_merged_bins[idx, :]
        
    # 对于每个细胞核，累加其所有像素点的基因表达向量，并计算像素点数
    for xi, yi in zip(x, y):
        idx = int(math.floor(xi) * math.ceil(patchsizey) + math.floor(yi))
        nucleus_expression[cell] += all_exp_merged_bins[idx, :]
        nucleus_pixel_count[cell] += 1

# 归一化细胞核的基因表达向量
for cell in nucleus_expression:
    if nucleus_pixel_count[cell] > 0:
        nucleus_expression[cell] /= nucleus_pixel_count[cell]

# 第三步：场分割的细胞分割
background_threshold = 0.1  # 概率小于此值的点视为背景点
cell_threshold = 0.15  # 概率大于此值的点视为细胞点
R=2
max_radius = 28.5  # 设置最大的细胞核半径

for i, (x, y, prob, cell) in enumerate(task2_result):
    if prob < background_threshold or cell != 0:
        continue

    # 计算到每个质心的距离
    field_vectors = np.array([centroids[c][:2] - [x, y] for c in centroids])
    distances = np.linalg.norm(field_vectors, axis=1)

    # 过滤距离超过max_radius的细胞核
    valid_indices = distances < max_radius
    valid_field_vectors = field_vectors[valid_indices]
    valid_centroids = [c for c, dist in zip(centroids, valid_indices) if dist]

    # 计算场力向量
    field_vectors = np.array([1 / dist**2 * similarity_expression_vectors(c, x, y, R) * vec
                              for c, vec, dist in zip(valid_centroids, valid_field_vectors, distances[valid_indices])])

    # 计算合力向量
    total_field_vector = np.sum(field_vectors, axis=0)

    # 判断点的归属
    if np.linalg.norm(total_field_vector) > cell_threshold:
        # 计算合力方向与每个有效场向量方向之间的角度
        angles = [angle_between(total_field_vector, vec) for vec in valid_field_vectors]
        closest_cell_index = np.argmin(angles)
        task2_result[i, 3] = valid_centroids[closest_cell_index]

# 更新结果
np.savetxt('results/updated_task2_result.txt', task2_result, fmt='%d')

image_width = 400  # 设置图像的宽度
image_height = 400  # 设置图像的高度

# 创建一个空的图像，用背景色填充
segmentation_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)  # 使用RGB格式

# 遍历每个像素点，为每个细胞核上的像素点分配不同的颜色
unique_cells = np.unique(task2_result[:, 3])
num_colors = int(max(unique_cells)) + 1  # 转换为整数
colors = plt.cm.jet(np.linspace(0, 1, num_colors))
# 确保颜色数组足够长以覆盖所有唯一的细胞核标签
if len(colors) <= max(unique_cells):
    colors = plt.cm.jet(np.linspace(0, 1, max(unique_cells) + 1))
for cell in unique_cells:
    if cell == 0:  # 跳过背景
        continue
    cell_index = int(cell)
    if cell_index >= len(colors):
        continue
    mask = task2_result[:, 3] == cell
    x, y = task2_result[mask, :2].T
    x, y = x.astype(int), y.astype(int)
    # 逐个像素点设置颜色，交换x和y的位置
    for xi, yi in zip(x, y):
        segmentation_image[xi, yi] = colors[cell_index][:3] * 255
# 显示和保存图像
plt.imshow(segmentation_image)
plt.axis('off')  # 不显示坐标轴
plt.savefig('results/final_result.png', bbox_inches='tight', pad_inches=0)


# 提取每个细胞核的大小（即像素个数）
cell_sizes = []
for cell in unique_cells:
    if cell == 0:  # 跳过背景
        continue
    mask = task2_result[:, 3] == cell
    cell_size = np.sum(mask)
    cell_sizes.append(cell_size)
plt.figure()
# 绘制细胞大小的分布图
plt.hist(cell_sizes, bins=50, color='blue', alpha=0.7)
plt.title('Cell Size Distribution')
plt.xlabel('Cell Size (number of pixels)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('fig/cell_size_distribution.png')


def angle_between(v1, v2):
    """计算两个向量之间的角度"""
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle

def similarity_expression_vectors(cell, x, y, R):
    idx = int(math.floor(x) * math.ceil(patchsizey) + math.floor(y))
    point_expression = all_exp_merged_bins[idx, :]

    # 使用之前计算的细胞核表达数据
    cell_expression = nucleus_expression[cell]

    # 计算相似性
    # 添加微小的噪声来避免完全的零向量
    noise = np.random.normal(0, 1e-9, len(cell_expression))
    cell_expression_noisy = cell_expression + noise
    point_expression_noisy = point_expression + noise

    # 计算皮尔逊相关系数
    correlation = spearmanr(cell_expression_noisy, point_expression_noisy)[0, 1]

    return correlation ** R

