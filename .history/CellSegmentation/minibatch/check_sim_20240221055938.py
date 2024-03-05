import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,pearsonr



if __name__=="__main__":
    
    # 获取当前脚本的绝对路径
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    cell_directory = os.path.dirname(script_directory)
    os.chdir(cell_directory)
    
    expression_matrix = np.load('dataset/expression_matrix.npy')
    
    task2_result = np.loadtxt('results/task2_result.txt')
    # 第二步：计算每个细胞核的加权质心、总基因表达
    unique_cells = np.unique(task2_result[:, 3])
    # 创建 cell 质心的空字典
    centroids = {cell: [0, 0, 0] for cell in unique_cells if cell != 0} 
    # 创建  cell 总基因表达的空字典
    nucleus_expression = {cell: np.zeros(expression_matrix.shape[2]) for cell in centroids if cell != 0}
    # 创建 cell 像素个数的空字典
    nucleus_pixel_count = {cell: 0 for cell in centroids if cell != 0}
    for cell in unique_cells:
        if cell == 0:
            continue
        mask = task2_result[:, 3] == cell
        
        x, y, prob = task2_result[mask, :3].T
        total_prob = np.sum(prob)
        if total_prob > 0:  # 检查 total_prob 是否大于0
            centroids[cell] = [np.sum(x * prob) / total_prob, np.sum(y * prob) / total_prob, total_prob]
        else:
            continue  # 如果 total_prob 为0，则跳过当前细胞核
        
        # 重置细胞核的总基因表达向量为0
        nucleus_expression[cell] = np.zeros(expression_matrix.shape[2])
        
        # 计算细胞核的总基因表达向量
        for xi, yi in zip(x.astype(int), y.astype(int)):
            # 直接使用(xi, yi)索引三维数组
            if 0 <= xi < expression_matrix.shape[0] and 0 <= yi < expression_matrix.shape[1]:
                nucleus_expression[cell] += expression_matrix[xi, yi, :]
            nucleus_pixel_count[cell] += 1

    # 归一化细胞核的基因表达向量
    for cell in nucleus_expression:
        if nucleus_pixel_count[cell] > 0:
            nucleus_expression[cell] /= nucleus_pixel_count[cell]
    
    np.savez_compressed('nucleus_expression.npz', **nucleus_expression)
    
    x=30
    for y in range(130,160):
        point_expression = expression_matrix[x,y,:]
        cell_expression = nucleus_expression[160]
        noise = np.random.normal(0, 1e-13, len(cell_expression))
        point_expression_noisy = point_expression + noise

        correlation, p_value = spearmanr(cell_expression, point_expression_noisy)

        if correlation <= 0:
            continue
        print(correlation)
            
        
        
        