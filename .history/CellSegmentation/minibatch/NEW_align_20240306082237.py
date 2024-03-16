import numpy as np
import scipy.ndimage as ndi
import os

def load_data():
    """
    从文本文件中加载细胞核概率数据和Watershed标签。

    Returns:
    task1_result: 细胞核概率数据。
    watershed: Watershed标签数据。
    """
    task1_result = np.loadtxt("dataset/task1_result.txt")
    watershed = np.loadtxt("dataset/watershed_labels.txt")
    return task1_result, watershed


def process_probabilities(task1_result, threshold):
    """
    处理概率数据，计算高于给定阈值的点的数量。

    Parameters:
    task1_result: 细胞核概率数据。
    threshold: 概率阈值。

    Returns:
    num_above_threshold: 高于阈值的点的数量。
    """
    probabilities = task1_result[:, 2]
    num_above_threshold = np.sum(probabilities >= threshold)
    return num_above_threshold


def create_matrices(task1_result, watershed):
    """
    根据概率数据和Watershed标签创建矩阵。

    Parameters:
    task1_result: 细胞核概率数据。
    watershed: Watershed标签数据。

    Returns:
    task1_matrix: 概率矩阵。
    watershed_matrix: Watershed标签矩阵。
    """
    task1_matrix = np.zeros((1200, 1200))
    watershed_matrix = np.zeros((1200, 1200))

    for x, y, value in task1_result:
        task1_matrix[int(x), int(y)] = value
    for x, y, label in watershed:
        watershed_matrix[int(x), int(y)] = int(label)
    return task1_matrix, watershed_matrix


def label_and_compare(task1_matrix, watershed_matrix, threshold, min_cell_size):
    """
    标记连通区域，比较Watershed标签，并处理新细胞核。

    Parameters:
    task1_matrix: 概率矩阵。
    watershed_matrix: Watershed标签矩阵。
    threshold: 概率阈值。
    min_cell_size: 新细胞核的最小像素点数。

    Returns:
    final_labels: 最终的标签矩阵。
    """
    task1_binary = task1_matrix >= threshold
    task1_labeled, _ = ndi.label(task1_binary)
    final_labels = np.zeros_like(task1_labeled)

    for label_num in np.unique(task1_labeled):
        mask = task1_labeled == label_num
        overlap = watershed_matrix[mask].astype(int)
        most_common = np.bincount(overlap[overlap > 0]).argmax() if overlap.any() else 0

        if most_common:
            final_labels[mask] = most_common
        elif np.sum(mask) > min_cell_size:
            distance = ndi.distance_transform_edt(~mask)
            nearest_label = watershed_matrix.flat[np.argmin(distance)]
            final_labels[mask] = nearest_label
    return final_labels


def save_results(task1_result, final_labels, filename="results/task2_result.txt"):
    """
    保存最终的细胞核标签结果到文本文件。

    Parameters:
    task1_result: 原始的细胞核概率数据。
    final_labels: 最终的标签矩阵。
    filename: 结果文件的路径。
    """
    new_col=final_labels[
        task1_result[:, 0].astype(int), task1_result[:, 1].astype(int)
    ]
    # task1_result[:, 3] = final_labels[
    #     task1_result[:, 0].astype(int), task1_result[:, 1].astype(int)
    # ]
    task1_result=np.hstack((task1_result[:,:3],new_col.reshape(-1, 1)))
    np.savetxt(filename, task1_result, fmt="%f")


def main():
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    cell_directory = os.path.dirname(script_directory)
    os.chdir(cell_directory)
    
    # 加载数据
    task1_result, watershed = load_data()

    # 设置阈值和最小细胞核大小
    threshold = 0.01
    min_cell_size = 5

    # 处理概率数据
    num_above_threshold = process_probabilities(task1_result, threshold)
    print(f"Number of points above threshold: {num_above_threshold}")
    print(f"Total number of points: {len(task1_result)}")

    # 创建矩阵
    task1_matrix, watershed_matrix = create_matrices(task1_result, watershed)

    # 标记、比较和处理
    final_labels = label_and_compare(
        task1_matrix, watershed_matrix, threshold, min_cell_size
    )

    # 统计和保存结果
    save_results(task1_result, final_labels)

    # 计算和打印0和非0标签的数量
    labels = final_labels.flatten()
    num_zeros = np.sum(labels == 0)
    num_non_zeros = np.sum(labels != 0)
    print(f"Number of zeros: {num_zeros}")
    print(f"Number of non-zeros: {num_non_zeros}")


if __name__ == "__main__":
    main()
