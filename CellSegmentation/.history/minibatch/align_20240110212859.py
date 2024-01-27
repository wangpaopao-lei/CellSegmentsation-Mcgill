import numpy as np
import scipy.ndimage as ndi  # 修改了导入方式
import matplotlib.pyplot as plt


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
    x, y, value = map(int, row)
    task1_matrix[x,y] = value

for row in watershed:
    x, y, label1 = map(int, row)
    watershed_matrix[x, y] = label1

num_above_threshold = np.sum(task1_matrix >= threshold)
print(f"Number of points in task1_matrix above threshold: {num_above_threshold}")

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
    overlap = watershed_matrix[mask]
    most_common = np.bincount(overlap[overlap > 0]).argmax()

    if most_common:
        final_labels[mask] = most_common
    else:
        # 处理新细胞核
        if mask.sum() > min_cell_size:
            distance = ndi.distance_transform_edt(~mask)
            nearest_label = watershed_matrix[np.argmin(distance)]
            final_labels[mask] = nearest_label

# 将 final_labels 转换回原始格式
final_labels_flat = np.zeros((160000, 1))
for i, (x, y) in enumerate(task1_result[:, :2].astype(int)):
    final_labels_flat[i] = final_labels[x, y]

# 输出结果
task2_result = np.hstack((task1_result, final_labels_flat))
np.savetxt('results/task2_result.txt', task2_result, fmt='%d')



# labels = task2_result[:, 3]

# # 计算0和非0的分布
# num_zeros = np.sum(labels == 0)
# num_non_zeros = np.sum(labels != 0)

# print(f"Number of zeros: {num_zeros}")
# print(f"Number of non-zeros: {num_non_zeros}")



image_width = 400  # 设置图像的宽度
image_height = 400  # 设置图像的高度

# 创建一个空的图像，用背景色填充
segmentation_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)  # 使用RGB格式

# 遍历每个像素点，为每个细胞核上的像素点分配不同的颜色
unique_cells = np.unique(task2_result[:, 3])
colors = plt.cm.jet(np.linspace(0, 1, len(unique_cells)))  # 使用colormap生成颜色

for cell in unique_cells:
    if cell == 0:  # 背景点
        continue
    mask = task2_result[:, 3] == cell
    x, y = task2_result[mask, :2].T
    x = x.astype(int)
    y = y.astype(int)
    segmentation_image[y, x] = colors[cell][:3] * 255  # 转换颜色为RGB格式

# 显示和保存图像
plt.imshow(segmentation_image)
plt.axis('off')  # 不显示坐标轴
plt.savefig('results/align_result.png', bbox_inches='tight', pad_inches=0)


