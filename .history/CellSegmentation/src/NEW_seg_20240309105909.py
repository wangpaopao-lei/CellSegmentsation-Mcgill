import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from scipy.stats import spearmanr, pearsonr
from scipy.sparse import load_npz
from PIL import Image
import time


def angle_between(v1, v2):
    """计算两个向量之间的角度"""
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle


def similarity_expression_vectors(cell, exp, x, y, R, nucleus_expression):

    x, y = int(x), int(y)
    point_expression = exp[x, y, :]

    # 使用之前计算的细胞核表达数据
    cell_expression = nucleus_expression[cell]

    # 计算相似性，添加微小的噪声来避免完全的零向量
    noise = np.random.normal(0, 1e-13, len(cell_expression))
    point_expression_noisy = point_expression + noise

    correlation, p_value = spearmanr(cell_expression, point_expression_noisy)

    if correlation <= 0:
        correlation = 1e-15

    return correlation**R


if __name__ == "__main__":

    # 获取当前脚本的绝对路径
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    cell_directory = os.path.dirname(script_directory)
    os.chdir(cell_directory)

    print("loading data\n")
    expression_path = "sparse_matrix.npz"
    expression_matrix = load_npz(expression_path)

    # 第一步：读取 task2_result 数据
    nuclei = pd.read_csv('/mnt/md0/lei/projects/CellSegmentsation/CellSegmentation/results/task2_result.txt', sep=' ', header=None)
    nuclei[0]=nuclei[0].astype(int)
    nuclei[1]=nuclei[1].astype(int)
    nuclei[3]=nuclei[3].astype(int)
    # 第二步：计算每个细胞核的加权质心、总基因表达
    print("calculating exp and centroids of nucleus\n")
    unique_cells = np.unique(nuclei[3])

    
    # 创建 cell 质心的空字典
    centroids = {cell: [0, 0, 0] for cell in unique_cells if cell != 0}
    # 创建  cell 总基因表达的空字典
    nucleus_expression = {
        cell: np.zeros(expression_matrix.shape[1]) for cell in centroids if cell != 0
    }
    # 创建 cell 像素个数的空字典
    nucleus_pixel_count = {cell: 0 for cell in centroids if cell != 0}

    # TODO(1): 更新细胞核的基因表达的计算方式
    
    for cell in unique_cells:
        if cell == 0:
            continue
        mask = nuclei[:, 3] == cell

        x, y, prob = nuclei[mask, :3].T
        total_prob = np.sum(prob)
        if total_prob > 0:  # 检查 total_prob 是否大于0
            centroids[cell] = [
                np.sum(x * prob) / total_prob,
                np.sum(y * prob) / total_prob,
                total_prob,
            ]
        else:
            continue  # 如果 total_prob 为0，则跳过当前细胞核

        centroids[cell] = [
            np.sum(x * prob) / total_prob,
            np.sum(y * prob) / total_prob,
            total_prob,
        ]

        # 对于每个细胞核，累加其所有像素点的基因表达向量，并计算像素点数
        for xi, yi in zip(x, y):
            xi, yi = int(xi), int(yi)
            nucleus_expression[cell] += expression_matrix[xi, yi, :]
            nucleus_pixel_count[cell] += 1

    # 归一化细胞核的基因表达向量
    for cell in nucleus_expression:
        if nucleus_pixel_count[cell] > 0:
            nucleus_expression[cell] /= nucleus_pixel_count[cell]

    # 第三步：场分割的细胞分割
    print("executing field segmentation\n")
    background_threshold = 0  # 概率小于此值的点视为背景点
    cell_threshold = 0.00001  # 合力大于此值的点视为细胞点
    R = 1
    max_radius = 28  # 设置最大的细胞半径

    total_field_magnitudes = []
    markers = []
    # test
    od = []
    line_count=0
    start_time=time.time()
    for i, (x, y, prob, cell) in enumerate(nuclei):
        line_count+=1
        if line_count %10000==0:
            current_time=time.time()
            elapsed_time = current_time - start_time
            print(f"{int(line_count/10000)} /144 batches completed! Time for batch: {elapsed_time:.2f} seconds")
            start_time = current_time
        # if prob <= background_threshold or cell != 0:
        #     continue
        if cell != 0:
            continue

        spot2centroid = np.array(
            [np.array(centroids[c][:2]) - np.array([x, y]) for c in centroids]
        )

        distances = np.linalg.norm(spot2centroid, axis=1)

        # 现在可以正确执行除法操作，因为 distances 能够广播到 field_vectors 的每一行
        normalized_field_vectors = spot2centroid / distances.reshape(-1, 1)

        # 过滤距离超过max_radius的细胞核
        valid_indices = distances < max_radius

        if np.sum(valid_indices) == 0:
            nuclei[i, 3] = 0
            continue

        valid_field_vectors = normalized_field_vectors[valid_indices]

        if np.sum(valid_indices) == 0:
            nuclei[i, 3] = 0
            continue

        valid_field_vectors = normalized_field_vectors[valid_indices]
        valid_centroids = [c for c, dist in zip(centroids, valid_indices) if dist]

        # 计算场力向量
        field_vectors = []
        mt1 = 0
        ms = 0
        md = 0

        for c, vec, dist in zip(
            valid_centroids, valid_field_vectors, distances[valid_indices]
        ):
            # weight_factor = 1 / dist**2
            weight_factor = 1 / (dist**2)
            similarity_measure = similarity_expression_vectors(
                c, expression_matrix, x, y, R, nucleus_expression
            )

            force_value = weight_factor * similarity_measure
            field_vector = force_value * vec

            if force_value > mt1:
                mt1 = force_value
                ms = similarity_measure
                md = weight_factor
            field_vectors.append(field_vector)

        # 提取最大力
        field_vectors = np.array(field_vectors)

        # print(field_vectors)
        # 计算合力向量
        total_field_vector = np.sum(field_vectors, axis=0)

        # 判断点的归属
        if mt1 > cell_threshold:
            # 计算合力方向与每个有效场向量方向之间的角度
            angles = [
                angle_between(total_field_vector, vec) for vec in valid_field_vectors
            ]
            closest_cell_index = np.argmin(angles)
            nuclei[i, 3] = valid_centroids[closest_cell_index]

        # test
        if (
            cell_threshold < mt1 < cell_threshold + cell_threshold / 8
            and nuclei[i, 3] == 54
        ):
            markers.append((x, y))
            od.append((x, y))
            print(f"X:{x}\tY:{y}\ttotal:{mt1}\t sim:{ms}\t dis:{md}")

    # 更新结果
    np.savetxt("results/updated_task2_result.txt", nuclei, fmt="%d")

    
    print("drawing figures\n")
    image_width = 1200  # 设置图像的宽度
    image_height = 1200  # 设置图像的高度
    # 创建一个空的图像，用背景色填充
    segmentation_image = np.zeros(
        (image_height, image_width, 3), dtype=np.uint8
    )  # 使用RGB格式
    # 遍历每个像素点，为每个细胞核上的像素点分配不同的颜色
    unique_cells = np.unique(nuclei[:, 3])
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
        mask = nuclei[:, 3] == cell
        x, y = nuclei[mask, :2].T
        x, y = x.astype(int), y.astype(int)
        # 逐个像素点设置颜色，交换x和y的位置
        for xi, yi in zip(x, y):
            segmentation_image[xi, yi] = colors[cell_index][:3] * 255

    # 提取每个细胞核的大小（即像素个数）
    cell_sizes = []
    for cell in unique_cells:
        if cell == 0:  # 跳过背景
            continue
        mask = nuclei[:, 3] == cell
        cell_size = np.sum(mask)
        cell_sizes.append(cell_size)
    print(f"Average size: {np.mean(cell_sizes)}")
    print(f"Std: {np.std(cell_sizes)}")

    # 显示和保存图像
    plt.figure()
    plt.imshow(segmentation_image)
    plt.axis("off")  # 不显示坐标轴
    for cell in unique_cells:
        if cell == 0:  # Skip background
            continue
        mask = nuclei[:, 3] == cell
        x, y = nuclei[mask, :2].T
        x_center, y_center = np.mean(x).astype(int), np.mean(y).astype(int)
        plt.text(
            y_center,
            x_center,
            str(cell),
            color="white",
            fontsize=4,
            ha="center",
            va="center",
        )
    for x, y in markers:
        plt.scatter(y, x, color="lime", s=20)
    plt.savefig("results/final_result.png", bbox_inches="tight", pad_inches=0)
    # # 将 NumPy 数组转换为 PIL 图像
    # segmentation_pil = Image.fromarray(segmentation_image)

    # # 保存图像
    # segmentation_pil.save('results/final_result.png')

    plt.figure()
    # 绘制细胞大小的分布图
    plt.hist(cell_sizes, bins=50, color="blue", alpha=0.7)
    plt.title("Cell Size Distribution")
    plt.xlabel("Cell Size (number of pixels)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("fig/cell_size_distribution.png")

    total_field_magnitudes = np.array(total_field_magnitudes)

    plt.figure()
    plt.hist(total_field_magnitudes, bins=50, color="blue", edgecolor="black")
    plt.title("Frequency Distribution of Total Field Vector Magnitudes")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")
    plt.savefig("fig/field_vector_distribution.png")
