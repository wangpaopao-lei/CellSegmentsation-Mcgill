#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/28 04:23
# @Author  : WangLei
# @File    : preprocess.py
# @Description :
import spateo as st
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import pickle
import os
from scipy.sparse import lil_matrix, csr_matrix, vstack
from scipy.ndimage import distance_transform_edt
from skimage import io, segmentation, color


def read_data(bin_file, image_file):
    """
    read transcripts data and image data
    """
    return st.io.read_bgi_agg(bin_file, image_file)


def crop_image_data(adatasub, start_x, start_y, end_x, end_y):
    """
    crop a smaller dataset for validation
    """
    adatasub.layers["unspliced"] = adatasub.X
    patchsizex = adatasub.X.shape[0]
    patchsizey = adatasub.X.shape[1]

    cropped_image_data = adatasub.layers["stain"][start_y:end_y, start_x:end_x]

    empty_X = np.zeros((end_y - start_y, end_x - start_x))
    adatasub = ad.AnnData(X=empty_X, uns=adatasub.uns.copy())
    adatasub.layers["stain"] = cropped_image_data
    return adatasub


def segment_nucleus(adatasub):
    """
    segment nucleus and save mask images
    """
    st.cs.mask_nuclei_from_stain(adatasub, otsu_classes=4, otsu_index=1)
    st.cs.find_peaks_from_mask(adatasub, "stain", 7)
    st.cs.watershed(adatasub, "stain", 5, out_layer="watershed_labels")

    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
    st.pl.imshow(adatasub, "watershed_labels", labels=True, alpha=0.5, ax=ax)
    plt.savefig("fig/watershed_result.png")

    with open("dataset/watershed_labels.txt", "w") as fw:
        for i in range(adatasub.layers["watershed_labels"].shape[0]):
            for j in range(adatasub.layers["watershed_labels"].shape[1]):
                fw.write(
                    str(i)
                    + "\t"
                    + str(j)
                    + "\t"
                    + str(adatasub.layers["watershed_labels"][i][j])
                    + "\n"
                )

    return adatasub


def calculate_distance_map(adatasub):
    """
    calculate and add watershed_distance layer
    """
    labels = adatasub.layers["watershed_labels"]
    unique_labels = np.unique(labels)[1:]  # 移除背景标签
    probability_map = np.zeros_like(labels, dtype=float)

    for label in unique_labels:
        binary_cell = labels == label
        boundaries = segmentation.find_boundaries(binary_cell, mode="inner")
        masked_boundaries = np.where(binary_cell, boundaries, 1)
        distance = distance_transform_edt(1 - masked_boundaries)
        max_distance = distance.max()

        if max_distance == 0:
            probability_map[labels == label] = 0.5  # 或其他固定值
        else:
            normalized_distance = distance / max_distance
            probability_map[labels == label] = normalized_distance[labels == label]
    adatasub.layers["watershed_distance"] = probability_map

    return adatasub


def preproccess_gene_data(
    bin_file, patchsizex, patchsizey, start_x, start_y, end_x, end_y
):
    """
    process bin data

    patchsizex and y mean the size of the original image e.g. 1200*1200
    adatasub.X.shape() means the size of cropped image e.g. 400*400
    """
    # find all the genes in the range
    geneid = {}
    genecnt = 0
    id2gene = {}
    with open(bin_file) as fr:
        header = fr.readline()
        for line in fr:
            gene, x, y, count = line.split()
            if gene not in geneid:
                geneid[gene] = genecnt
                id2gene[genecnt] = gene
                genecnt += 1

    # 这一步目的是得到一维索引到 gene 的稀疏矩阵
    idx2exp = {}
    with open(bin_file) as fr:
        header = fr.readline()
        for line in fr:
            gene, x, y, count = line.split()
            if gene not in geneid:
                continue

            idx = int(math.floor(int(x)) * math.ceil(patchsizey) + math.floor(int(y)))

            if idx not in idx2exp:
                idx2exp[idx] = {}
                idx2exp[idx][geneid[gene]] = int(count)
            elif geneid[gene] not in idx2exp[idx]:
                idx2exp[idx][geneid[gene]] = int(count)
            else:
                idx2exp[idx][geneid[gene]] += int(count)

    all_exp_merged_bins = lil_matrix(
        (int(math.ceil(patchsizex) * math.ceil(patchsizey)), genecnt), dtype=np.int8
    )
    for idx in idx2exp:
        for gid in idx2exp[idx]:
            all_exp_merged_bins[idx, gid] = idx2exp[idx][gid]

    all_exp_merged_bins = all_exp_merged_bins.tocsr()
    # 创建 anndata 是为了应用选取高变基因的函数
    all_exp_merged_bins_ad = ad.AnnData(
        all_exp_merged_bins,
        obs=pd.DataFrame(index=[i for i in range(all_exp_merged_bins.shape[0])]),
        var=pd.DataFrame(index=[i for i in range(all_exp_merged_bins.shape[1])]),
    )
    sc.pp.highly_variable_genes(
        all_exp_merged_bins_ad, n_top_genes=2000, flavor="seurat_v3", span=1.0
    )
    selected_index = all_exp_merged_bins_ad.var[
        all_exp_merged_bins_ad.var.highly_variable
    ].index
    selected_index = list(selected_index)
    selected_index = [int(i) for i in selected_index]
    # 将稀疏矩阵转换为数组，由一维坐标索引
    all_exp_merged_bins = all_exp_merged_bins.toarray()[:, selected_index]

    # 转换为二维坐标索引的数组
    expression_matrix = np.zeros(
        (end_y - start_y, end_x - start_x, len(selected_index))
    )
    # 想得到一个与裁剪后图像大小相同的基因表达数组
    for idx, expression in enumerate(all_exp_merged_bins):
        y = idx % math.ceil(patchsizey)
        x = idx // math.ceil(patchsizey)
        # 确保 x, y 在图像尺寸范围内
        if start_x <= x < end_x and start_y <= y < end_y:
            # 需要对坐标进行偏移
            expression_matrix[y - start_y, x - start_x, :] = expression

    print(expression_matrix.shape)
    np.save("dataset/expression_matrix.npy", expression_matrix)

    return expression_matrix


def calculate_offsets(distance_range):
    """
    计算给定距离范围内的所有偏移量。
    """
    offsets = []
    for dis in range(1, distance_range + 1):
        offsets += [[-dis, dy] for dy in range(-dis, dis + 1)]
        offsets += [[dis, dy] for dy in range(-dis, dis + 1)]
        offsets += [[dx, -dis] for dx in range(-dis + 1, dis)]
        offsets += [[dx, dis] for dx in range(-dis + 1, dis)]
    return offsets


def prepare_samples(layers, shape, expression_matrix, offsets, threshold, n_neighbor,downrs):
    """准备训练和测试样本。"""
    x_train, x_pos, x_labels, y_train = [], [], [], []
    x_train_outer, x_pos_outer, x_labels_outer, y_train_outer = [], [], [], []
    x_test, x_test_pos, x_test_labels = [], [], []
    x_test_tmp=[]
    test_batch_size=10000
    test_batch_count=0
    counter_core, counter_outer = 0, 0  # 计数器
    
    temp_dir = 'temp_x_test_batches'
    try:
        os.mkdir(temp_dir)
    except FileExistsError:
        print('temp folder exists')
    
    for i in range(shape[0]):
        if (i + 1) % 100 == 0:
            print("finished {0:.0%}".format(i / layers["watershed_labels"].shape[0]))
        for j in range(shape[1]):

            label, distance = (
                layers["watershed_labels"][i, j],
                layers["watershed_distance"][i, j],
            )

            samples, pos, labels = process_sample(
                i, j, distance, offsets, layers, expression_matrix, shape, n_neighbor
            )

            if label > 0 and distance >= threshold and i%downrs==0 and j%downrs==0:
                counter_core += 1
                x_train.append(samples)
                x_pos.append(pos)
                x_labels.append(labels)
                y_train.append(1)
            elif label == 0 and distance >= threshold and i%downrs==0 and j%downrs==0:
                counter_outer += 1
                x_train_outer.append(samples)
                x_pos_outer.append(pos)
                x_labels_outer.append(labels)
                y_train_outer.append(0)
            
            
            x_test_tmp.append(samples)
            if len(x_test_tmp)>=test_batch_size:
                batch_path = os.path.join(temp_dir, f'x_test_batch_{test_batch_count}.npz')
                np.savez_compressed(batch_path, x_test=np.array(x_test_tmp))
                x_test_tmp = []  # 清空列表以释放内存
                test_batch_count += 1
                
            x_test_pos.append(pos)
            x_test_labels.append(labels)

    if x_test_tmp:
        batch_path = os.path.join(temp_dir, f'x_test_batch_{test_batch_count}.npz')
        np.savez_compressed(batch_path, x_test=np.array(x_test_tmp))
    
    num_samples = min(len(x_train), len(x_train_outer))
    random_indices = np.random.choice(len(x_train_outer), num_samples, replace=False)
    x_train = x_train + [x_train_outer[i] for i in random_indices]
    x_pos = x_pos + [x_pos_outer[i] for i in random_indices]
    x_labels = x_labels + [x_labels_outer[i] for i in random_indices]
    y_train = y_train + [y_train_outer[i] for i in random_indices]

    return (
        x_train,
        x_pos,
        x_labels,
        y_train,
        x_test,
        x_test_pos,
        x_test_labels,
        counter_core,
        counter_outer,
    )


def process_sample(
    i, j, distance, offsets, layers, expression_matrix, shape, n_neighbor
):
    """
    处理单个样本，收集其周围空间位置的基因表达数据。

    参数:
    - i, j: 当前位置的行列索引。
    - label: 当前位置的标签。
    - distance: 当前位置到最近核心的距离。
    - offsets: 周围空间位置的偏移量列表。
    - is_core: 表示当前样本是否属于核心区域。
    - all_exp_merged_bins: 基因表达数据矩阵。
    - adatasub: 包含标签和距离信息的anndata对象。
    - n_neighbor: 要考虑的周围邻居数量。
    - patchsizey: 图像在y方向上的尺寸。

    返回:
    - sample: 收集到的基因表达数据。
    - pos: 对应的空间位置列表。
    - labels: 每个位置的距离标签。
    """
    samples = [expression_matrix[i, j, :]]  # 当前位置的基因表达数据
    pos = [[i, j]]  # 当前位置
    labels = [distance]  # 当前位置的距离标签

    for dx, dy in offsets:

        x, y = i + dx, j + dy
        if 0 <= x < shape[0] and 0 <= y < shape[1]:

            samples.append(expression_matrix[x, y, :])
            pos.append([x, y])
            labels.append(layers["watershed_distance"][x, y])
        if len(samples) >= n_neighbor:
            break

    return np.array(samples), np.array(pos), np.array(labels)

def merge_batches(temp_dir, final_file):
    batches = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.startswith('x_test_batch_')]
    all_data = []
    for batch_file in sorted(batches):
        data = np.load(batch_file)['x_test']
        all_data.append(data)
    # 合并所有批次的数据
    all_data = np.concatenate(all_data, axis=0)
    # 保存到最终的文件
    np.savez_compressed(final_file, x_test=all_data)
    # 清理临时文件
    for batch_file in batches:
        os.remove(batch_file)
    os.rmdir(temp_dir)

if __name__ == "__main__":

    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    cell_directory = os.path.dirname(script_directory)
    os.chdir(cell_directory)

    bin_file = "data/Mouse_brain_Adult_GEM_bin1_sub.tsv"
    image_file = "data/Mouse_brain_Adult_sub.tif"
    
    start_x = 0
    start_y=0
    end_x = 1200
    end_y=1200
    distance_ratio_threshold = 0.35
    n_neighbor = 30

    adatasub = read_data(bin_file, image_file)

    patchsizex = adatasub.X.shape[0]
    patchsizey = adatasub.X.shape[1]

    adatasub = crop_image_data(adatasub, start_x, start_y, end_x, end_y)
    adatasub = segment_nucleus(adatasub)
    adatasub = calculate_distance_map(adatasub)
    
    print("watershed labels prepared.")
    
    expression_matrix = preproccess_gene_data(
        bin_file, patchsizex, patchsizey, start_x, start_y, end_x, end_y
    )
    
    print("gene data prepared.")
    
    offsets = calculate_offsets(10)
    
    (
        x_train,
        x_train_pos,
        x_train_labels,
        y_train,
        x_test,
        x_test_pos,
        x_test_labels,
        counter_core,
        counter_outer,
    ) = prepare_samples(
        adatasub.layers,
        adatasub.layers["watershed_labels"].shape,
        expression_matrix,
        offsets,
        distance_ratio_threshold,
        n_neighbor,
        downrs=2
    )
    # 打印计数器结果
    print(f"Core samples: {counter_core}, Outer samples: {counter_outer}")

    np.savez_compressed("dataset/x_train.npz", x_train=x_train)
    np.savez_compressed("dataset/x_train_pos.npz", x_train_pos=x_train_pos)
    np.savez_compressed("dataset/x_train_labels.npz", x_train_labels=x_train_labels)
    np.savez_compressed("dataset/y_train.npz", y_train=y_train)

    
    merge_batches('temp_x_test_batches', 'dataset/x_test.npz')
    # np.savez_compressed("dataset/x_test.npz", x_test=x_test)
    np.savez_compressed("dataset/x_test_pos.npz", x_test_pos=x_test_pos)
    np.savez_compressed("dataset/x_test_labels.npz", x_test_labels=x_test_labels)
    
    print("done!")
