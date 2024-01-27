import numpy as np
from scipy.stats import pearsonr

def find_overlapping_cells(r1, r2):
    unique_cells_r1 = np.unique(r1[r1 > 0])
    overlapping_cells = {}

    for cell_r1 in unique_cells_r1:
        mask_r1 = r1 == cell_r1
        overlapping_cells_r2 = r2[mask_r1]
        unique, counts = np.unique(overlapping_cells_r2, return_counts=True)
        if len(unique) > 1 or (len(unique) == 1 and unique[0] != 0):
            best_match = unique[np.argmax(counts)]
            if best_match != 0:  # Exclude background
                overlapping_cells[cell_r1] = best_match

    return overlapping_cells


def calculate_expression_profile(r, mask, all_exp_merged_bins):
    indices = np.argwhere(mask)
    expressions = [all_exp_merged_bins[int(i * patchsizey + j), :] for i, j in indices if r[i, j] > 0]
    if expressions:
        return np.mean(expressions, axis=0)
    else:
        return np.zeros(all_exp_merged_bins.shape[1])


with np.load('dataset/expression_data.npz') as data:
    all_exp_merged_bins = data['expression_data']
patchsizey=1200

start_x, start_y = 800, 0  # 根据图像实际情况调整
end_x, end_y = 1200, 400
# 初始化一个400x400的二维数组
r1 = np.zeros((400, 400), dtype=int)
# 打开并读取结果文件
with open('results/spot2cell_0:0:0:0.txt', 'r') as file:
    for line in file:
        # 解析每行的坐标和标签
        parts = line.strip().split('\t')  # 使用 strip() 移除尾部的换行符
        coordinates = parts[0]
        label = int(float(parts[1]))  # 先将标签转换为浮点数，然后转换为整数

        x, y = map(int, coordinates.split(':'))

        # 检查点是否在右上角400x400范围内
        if start_x <= x < end_x and start_y <= y < end_y:
            # 转换坐标系统并保存到数组中
            r1[y - start_y, x - start_x] = label

r2 = np.zeros((400, 400), dtype=int)
# 读取第二份结果文件
task2_result = np.loadtxt('results/updated_task2_result.txt')

# 遍历结果数据，将每个点的分割标签存储到二维数组中
for row in task2_result:
    x, y, label = int(row[0]), int(row[1]), int(row[3])  # 假设标签在第四列
    r2[y, x] = label  # 假设文件中的坐标是基于0的索引

# 假设 all_exp_merged_bins 已加载
overlapping_cells = find_overlapping_cells(r1, r2)
patchsizey = 1200  # 或根据实际数据调整

correlations_xc_xint = []
correlations_x0c_xint = []
for cell_r1, cell_r2 in overlapping_cells.items():
    mask_c = r1 == cell_r1
    mask_c0 = r2 == cell_r2
    mask_xint = mask_c & mask_c0
    mask_xc = mask_c & ~mask_c0
    mask_x0c = mask_c0 & ~mask_c

    xint_expression = calculate_expression_profile(r1, mask_xint, all_exp_merged_bins)
    xc_expression = calculate_expression_profile(r1, mask_xc, all_exp_merged_bins)
    x0c_expression = calculate_expression_profile(r2, mask_x0c, all_exp_merged_bins)

    correlation_xc_xint = pearsonr(xc_expression, xint_expression)[0]
    correlation_x0c_xint = pearsonr(x0c_expression, xint_expression)[0]

    # 忽略nan值
    if not np.isnan(correlation_xc_xint):
        correlations_xc_xint.append(correlation_xc_xint)
    if not np.isnan(correlation_x0c_xint):
        correlations_x0c_xint.append(correlation_x0c_xint)

# 计算平均相关性
if correlations_xc_xint:  # 确保列表不为空
    avg_correlation_xc_xint = np.mean(correlations_xc_xint)
else:
    avg_correlation_xc_xint = 0  # 或其他适当的默认值

if correlations_x0c_xint:  # 确保列表不为空
    avg_correlation_x0c_xint = np.mean(correlations_x0c_xint)
else:
    avg_correlation_x0c_xint = 0  # 或其他适当的默认值

# 输出平均相关性
print(f"Average Correlation xc-xint: {avg_correlation_xc_xint}")
print(f"Average Correlation x0c-xint: {avg_correlation_x0c_xint}")







