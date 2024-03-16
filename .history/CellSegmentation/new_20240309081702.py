from scipy.sparse import load_npz
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix

def xy2idx_vectorized(x, y):
    return (np.floor(x).astype(int) * np.ceil(1200) + np.floor(y).astype(int)).astype(int)


if __name__ == "__main__":

    # 获取当前脚本的绝对路径
    script_path = os.path.abspath(__file__)

    cell_directory = os.path.dirname(script_path)
    os.chdir(cell_directory)
    print(cell_directory)
    # 加载稀疏矩阵
    sparse_matrix = load_npz('dataset/full_gene_sparse_matrix.npz')

    # 转换为密集矩阵
    dense_matrix = sparse_matrix.todense()

    # 创建 DataFrame
    df = pd.DataFrame(dense_matrix)
    nuclei = pd.read_csv('/mnt/md0/lei/projects/CellSegmentsation/CellSegmentation/results/task2_result.txt', sep=' ', header=None)
    nuclei[0]=nuclei[0].astype(int)
    nuclei[1]=nuclei[1].astype(int)
    nuclei[3]=nuclei[3].astype(int)

    n=5
    result_df = pd.DataFrame(0, index=df.index, columns=df.columns)

    offsets = np.array([(dx, dy) for dx in range(-(n//2), n//2 + 1) for dy in range(-(n//2), n//2 + 1)])
    for idx,row in df.iterrows():
        
        neighbor_coords = np.array([x, y]) + offsets
        # 使用向量化的xy2idx计算所有邻域的索引
        neighbor_idxs = xy2idx_vectorized(neighbor_coords[:, 0], neighbor_coords[:, 1])
        
        # 检查索引有效性并累加向量
        sum_vector = np.zeros(len(row))
        for neighbor_idx in neighbor_idxs:
            if neighbor_idx in df.index:  # 确保索引存在于df中
                sum_vector += df.loc[neighbor_idx].values

        # 更新结果DataFrame
        result_df.loc[idx] = sum_vector
    sparse_matrix = csr_matrix(result_df.values)
    save_npz('sparse_matrix.npz', sparse_matrix)

