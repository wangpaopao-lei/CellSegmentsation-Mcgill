import numpy as np

# 定义一个函数来检查和报告数据中的 NaN 和无穷大值
def check_data(file_path):
    try:
        # 加载 npz 文件
        data = np.load(file_path, allow_pickle=True)
        # 遍历文件中的每个数组
        for key in data:
            array = data[key].astype(np.float32)  # 确保数据是 float 类型以检查 NaN
            print(f"检查 {key} 中...")
            check_array(array)
    except Exception as e:
        print(f"加载数据时出错: {e}")

# 定义一个辅助函数来检查 NaN 和无穷大值
def check_array(array):
    nan_indices = np.argwhere(np.isnan(array))
    inf_indices = np.argwhere(np.isinf(array))

    if nan_indices.size > 0:
        print(f"发现 NaN 值的位置：{nan_indices}")
    else:
        print("数据中没有 NaN 值。")

    if inf_indices.size > 0:
        print(f"发现无穷大值的位置：{inf_indices}")
    else:
        print("数据中没有无穷大值。")

# 文件路径列表，需要替换成实际的文件路径
file_paths = [
    'dataset/x_test0:0:0:0.npz',
    'dataset/x_test_pos0:0:0:0.npz',
    'dataset/x_test_labels0:0:0:0.npz',
    # 'dataset/y_optimize_train0:0:0:0.npz',
    # 'dataset/y_bin0:0:0:0.npz',
    'dataset/watershed_distance.npz'
    # 添加任何其他需要检查的 npz 文件路径
]

# 对每个文件路径执行检查
for path in file_paths:
    print(f"正在检查文件：{path}")
    check_data(path)
