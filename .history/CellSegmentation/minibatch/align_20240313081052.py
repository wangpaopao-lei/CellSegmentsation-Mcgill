import numpy as np
import scipy.ndimage as ndi  # 修改了导入方式
import matplotlib.pyplot as plt
import os
if __name__ == "__main__":

    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    cell_directory = os.path.dirname(script_directory)
    os.chdir(cell_directory)
    # 可以修改的变量
    threshold = 0.01  # 用于确定细胞核的概率阈值
    min_cell_size = 5  # 新细胞核的最小像素点数

    # 读取数据
    task1_result = np.loadtxt('dataset/task1_result.txt')
    watershed = np.loadtxt('dataset/watershed_labels.txt')


    probabilities = task1_result[:, 2]
    # 计算大于阈值的点的数量
    num_above_threshold = np.sum(probabilities >= threshold)

    print(f"Number of points above threshold: {num_above_threshold}")
    print(f"Total number of points: {len(probabilities)}")


    # 将数据转换为 400x400 矩阵
    task1_matrix = np.zeros((400, 400))
    watershed_matrix = np.zeros((400, 400))

    for row in task1_result:
        x, y = map(int, row[:2])  # 只将x和y转换为整数
        value = row[2]  # 保持value为原始的浮点数
        task1_matrix[x, y] = value  # 注意交换x和y以匹配行和列的索引

    for row in watershed:
        x, y, label1 = map(int, row)
        watershed_matrix[x, y] = label1


    # 处理 task1_result
    task1_binary = task1_matrix >= threshold
    # 标记连通区域
    task1_labeled, num_features = ndi.label(task1_binary)


    # 初始化最终结果数组
    final_labels = np.zeros_like(task1_labeled)

    # 对比 task1_labeled 和 watershed 的标签
    for label_num in np.unique(task1_labeled):
        if label_num == 0:
            continue  # 跳过背景
        mask = task1_labeled == label_num
        overlap = watershed_matrix[mask].astype(int)  # 转换为整数
        if overlap.size > 0 and np.any(overlap > 0):
            most_common = np.bincount(overlap[overlap > 0]).argmax()
        else:
            most_common = 0  # 如果overlap为空或没有非零元素，设置默认值

        if most_common:
            final_labels[mask] = most_common
        else:
            # 处理新细胞核
            if mask.sum() > min_cell_size:
                distance = ndi.distance_transform_edt(~mask)
                nearest_label = watershed_matrix[np.argmin(distance)]
                final_labels[mask] = nearest_label


    # 计算细胞核的个数
    unique_nuclei = np.unique(final_labels[final_labels > 0])
    num_nuclei = len(unique_nuclei)
    print(f"Number of nuclei: {num_nuclei}")

    final_labels_flat = np.zeros((1440000,))
    for i, (x, y) in enumerate(task1_result[:, :2].astype(int)):
        final_labels_flat[i] = final_labels[x, y]

    # 输出结果
    # 使用 task1_result 的前三列（包含原始概率值）
    # 注意：这里不需要再使用 np.newaxis
    task2_result = np.hstack((task1_result[:, :3], final_labels_flat.reshape(-1, 1)))

    # 输出结果，保留概率值在第三列，细胞核标签在第四列
    np.savetxt('results/task2_result.txt', task2_result, fmt='%f')



    labels = task2_result[:, 3]

    # 计算0和非0的分布
    num_zeros = np.sum(labels == 0)
    num_non_zeros = np.sum(labels != 0)

    print(f"Number of zeros: {num_zeros}")
    print(f"Number of non-zeros: {num_non_zeros}")




