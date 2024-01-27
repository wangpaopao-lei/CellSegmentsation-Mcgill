import numpy as np

# 第一步：读取 task2_result 数据
task2_result = np.loadtxt('task2_result.txt')
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

for i, (x, y, prob, cell) in enumerate(task2_result):
    if prob < background_threshold:
        continue  # 跳过背景点
    if cell != 0:
        continue  # 跳过已标记的细胞核点

    field_vectors = np.array([centroids[c][:2] - [x, y] for c in centroids])
    distances = np.linalg.norm(field_vectors, axis=1)
    field_strengths = centroids.values()[:, 2] / distances  # 根据距离计算场力大小

    # 根据最大场力判断点的归属
    max_field = np.max(field_strengths)
    if max_field > cell_threshold:
        task2_result[i, 3] = unique_cells[np.argmax(field_strengths)]

# 更新结果
np.savetxt('updated_task2_result.txt', task2_result, fmt='%d')
