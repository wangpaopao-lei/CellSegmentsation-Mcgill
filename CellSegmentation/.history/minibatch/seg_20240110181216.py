import numpy as np

# 第一步：读取 task2_result 数据
task2_result = np.loadtxt('results/task2_result.txt')
# 假设 task2_result 的格式与前面一致

# 第二步：计算每个细胞核的加权质心
unique_cells = np.unique(task2_result[:, 3])
centroids = {cell: [0, 0, 0] for cell in unique_cells if cell != 0}  # 排除背景

for cell in unique_cells:
    if cell == 0:
        continue
    mask = task2_result[:, 3] == cell
    x, y, prob = task2_result[mask, :3].T
    total_prob = np.sum(prob)
    centroids[cell] = [np.sum(x * prob) / total_prob, np.sum(y * prob) / total_prob, total_prob]

# 第三步：场分割的细胞分割
background_threshold = 0.1  # 概率小于此值的点视为背景点
cell_threshold = 0.2  # 概率大于此值的点视为细胞点
R=1
max_radius = ...  # 设置最大的细胞核半径

for i, (x, y, prob, cell) in enumerate(task2_result):
    if prob < background_threshold or cell != 0:
        continue

    # 计算到每个质心的距离和场向量
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

def angle_between(v1, v2):
    """计算两个向量之间的角度"""
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle

def similarity_expression_vectors(cell, x, y, R):
    # 实现基因表达向量相似性计算
    pass

# 更新结果
np.savetxt('results/updated_task2_result.txt', task2_result, fmt='%d')

