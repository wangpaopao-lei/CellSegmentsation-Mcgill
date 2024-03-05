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
import os
from scipy.sparse import lil_matrix, csr_matrix, vstack
from scipy.ndimage import distance_transform_edt
from skimage import io,segmentation,color




# read file
bin_file = 'data/Mouse_brain_Adult_GEM_bin1_sub.tsv'
image_file = 'data/Mouse_brain_Adult_sub.tif'
adatasub = st.io.read_bgi_agg(bin_file, image_file)




bin_size = 2
n_neighbor = 30
r_estimate = 15
startx = '0'
starty = '0'
patchsize = '0'



# 裁剪区域的起始和结束坐标
starx, stary = 800, 0
endx, endy = 1200, 400
# 裁剪数据集
adatasub.layers['unspliced'] = adatasub.X
patchsizex = adatasub.X.shape[0]
patchsizey = adatasub.X.shape[1]
print("patchsizey: "+str(patchsizey))
# 裁剪图像数据
cropped_image_data = adatasub.layers['stain'][0:400, 800:1200]
# 创建一个空的 X 矩阵与裁剪图像相同的尺寸
empty_X = np.zeros((400, 400))
# 创建新的 AnnData 对象并复制必需的元数据
adatasub = ad.AnnData(X=empty_X, uns=adatasub.uns.copy())
# 添加裁剪后的图像数据到新对象的 layers
adatasub.layers['stain'] = cropped_image_data







# nucleus segmentation from staining image
fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
st.cs.mask_nuclei_from_stain(adatasub, otsu_classes=4, otsu_index=1)
st.pl.imshow(adatasub, 'stain_mask', ax=ax)
st.cs.find_peaks_from_mask(adatasub, 'stain', 7)
st.cs.watershed(adatasub, 'stain', 5, out_layer='watershed_labels')

fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
st.pl.imshow(adatasub, 'stain', save_show_or_return='return', ax=ax)
plt.savefig('fig/watershed_labels'+ startx + ':' + starty + ':' + '.png')
st.pl.imshow(adatasub, 'watershed_labels', labels=True, alpha=0.5, ax=ax)
plt.savefig('fig/watershed_result' + startx + ':' + starty + ':' + '.png')

with open('dataset/watershed_labels.txt','w') as fw:
    for i in range(adatasub.layers['watershed_labels'].shape[0]):
        for j in range(adatasub.layers['watershed_labels'].shape[1]):
            fw.write(str(i) + '\t' + str(j) + '\t' + str(adatasub.layers['watershed_labels'][i][j]) + '\n')
        
adatasub.write('data/Mouse_brain_Adult.h5ad')
# print(adatasub)
watershed_labels=adatasub.layers['watershed_labels']





# 获取唯一的标签值（除了背景）
unique_labels = np.unique(watershed_labels)
unique_labels = unique_labels[unique_labels != 0]  # 移除背景标签

probability_map = np.zeros_like(watershed_labels, dtype=float)

for label in unique_labels:
    binary_cell = (watershed_labels == label)
    boundaries = segmentation.find_boundaries(binary_cell, mode='inner')
    masked_boundaries = np.where(binary_cell, boundaries, 1)
    distance = distance_transform_edt(1-masked_boundaries)
    max_distance = distance.max()
    
    if max_distance == 0:
        probability_map[watershed_labels == label] = 0.5  # 或其他固定值
    else:
        normalized_distance = distance / max_distance
        probability_map[watershed_labels == label] = normalized_distance[watershed_labels == label]

adatasub.layers['watershed_distance']=probability_map

np.savez_compressed('dataset/watershed_distance',watershed_distance=probability_map)
adatasub.write('dataset/spots' + startx + ':' + starty + ':' + ':' + patchsize + ':' + patchsize + '.h5ad')






watershed2x = {}
watershed2y = {}
for i in range(adatasub.layers['watershed_labels'].shape[0]):
    for j in range(adatasub.layers['watershed_labels'].shape[1]):
        if adatasub.layers['watershed_labels'][i, j] == 0:
            continue
        if adatasub.layers['watershed_labels'][i, j] in watershed2x:
            watershed2x[adatasub.layers['watershed_labels'][i, j]].append(i)
            watershed2y[adatasub.layers['watershed_labels'][i, j]].append(j)
        else:
            watershed2x[adatasub.layers['watershed_labels'][i, j]] = [i]
            watershed2y[adatasub.layers['watershed_labels'][i, j]] = [j]

watershed2center = {}
sizes = []
for nucleus in watershed2x:
    watershed2center[nucleus] = [np.mean(watershed2x[nucleus]), np.mean(watershed2y[nucleus])]
    sizes.append(len(watershed2x[nucleus]))
# print(np.min(sizes), np.max(sizes), np.mean(sizes))
# print('#nucleus', len(watershed2center))






# find xmin ymin
xall = []
yall = []
with open(bin_file) as fr:
    header = fr.readline()
    for line in fr:
        gene, x, y, count = line.split()
        xall.append(int(x))
        yall.append(int(y))
xmin = np.min(xall)
ymin = np.min(yall)
print(np.min(xall), np.min(yall), np.max(xall), np.max(yall))

# find all the genes in the range
geneid = {}
genecnt = 0
id2gene = {}
with open(bin_file) as fr:
    header = fr.readline()
    for line in fr:
        gene, x, y, count = line.split()
        x, y = int(x) - xmin, int(y) - ymin
        if not (starx <= x < endx and stary <= y < endy):
            continue
        # 将坐标调整为裁剪后的图像坐标
        gene, x, y, count = line.split()
        if gene not in geneid:
            geneid[gene] = genecnt
            id2gene[genecnt] = gene
            genecnt += 1

idx2exp = {}
downrs = bin_size
with open(bin_file) as fr:
    header = fr.readline()
    for line in fr:
        gene, x, y, count = line.split()
        x, y = int(x) - xmin, int(y) - ymin
        if not (starx <= x < endx and stary <= y < endy):
            continue
        # 将坐标调整为裁剪后的图像坐标
        x -= starx
        y -= stary
        gene, x, y, count = line.split()
        x = int(x) - xmin
        y = int(y) - ymin
        if gene not in geneid:
            continue
        if int(x) < int(startx) or int(x) >= int(startx) + int(patchsizex) or int(y) < int(starty) or int(y) >= int(
                starty) + int(patchsizey):
            continue
        idx = int(math.floor((int(x) - int(startx))) * math.ceil(patchsizey) + math.floor(
            (int(y) - int(starty))))
        if idx not in idx2exp:
            idx2exp[idx] = {}
            idx2exp[idx][geneid[gene]] = int(count)
        elif geneid[gene] not in idx2exp[idx]:
            idx2exp[idx][geneid[gene]] = int(count)
        else:
            idx2exp[idx][geneid[gene]] += int(count)

all_exp_merged_bins = lil_matrix((int(math.ceil(patchsizex) * math.ceil(patchsizey)), genecnt),
                                 dtype=np.int8)
for idx in idx2exp:
    for gid in idx2exp[idx]:
        all_exp_merged_bins[idx, gid] = idx2exp[idx][gid]
        # print(idx, gid, idx2exp[idx][gid])
all_exp_merged_bins = all_exp_merged_bins.tocsr()
# print(all_exp_merged_bins.shape)

all_exp_merged_bins_ad = ad.AnnData(
    all_exp_merged_bins,
    obs=pd.DataFrame(index=[i for i in range(all_exp_merged_bins.shape[0])]),
    var=pd.DataFrame(index=[i for i in range(all_exp_merged_bins.shape[1])]),
)
sc.pp.highly_variable_genes(all_exp_merged_bins_ad, n_top_genes=2000, flavor='seurat_v3', span=1.0)
selected_index = all_exp_merged_bins_ad.var[all_exp_merged_bins_ad.var.highly_variable].index
selected_index = list(selected_index)
selected_index = [int(i) for i in selected_index]
with open('dataset/variable_genes' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.txt', 'w') as fw:
    for id in selected_index:
        fw.write(id2gene[id] + '\n')

# check total gene counts
all_exp_merged_bins = all_exp_merged_bins.toarray()[:, selected_index]
np.savez_compressed('dataset/expression_data.npz', expression_data=all_exp_merged_bins)






# 设定距离中心的比例阈值，例如选取最近的30%的点
distance_ratio_threshold = 0.35
counter_core = 0  # 核心部分的计数器
counter_outer = 0  # 核外部分的计数器

# 前面已经定义的变量
# downrs, offsets, all_exp_merged_bins, adatasub, n_neighbor, patchsizey

# 初始化变量
# x_test_tmp = []
# x_test = []
# x_test_pos = []
# x_test_dis= []
# x_test_dis_tmp=[]
label_img = np.stack((watershed_labels,) * 3, axis=-1)
dataset_map = np.ones_like(label_img,dtype=float)
train_color = [1, 0, 0]  # 红色
test_color = [0, 0, 1]   # 蓝色

x_train = []
x_pos = []
x_labels = []
y_train = []
y_bin = []
offsets = []

x_train_outer = []
x_pos_outer = []
x_labels_outer = []
y_train_outer = []
y_bin_outer = []

x_test = []
x_test_pos = []
x_test_labels = []

for dis in range(1, 11):
    for dy in range(-dis, dis + 1):
        offsets.append([-dis, dy])
    for dy in range(-dis, dis + 1):
        offsets.append([dis, dy])
    for dx in range(-dis + 1, dis):
        offsets.append([dx, -dis ])
    for dx in range(-dis + 1, dis):
        offsets.append([dx, dis])
for i in range(adatasub.layers['watershed_labels'].shape[0]):
    if (i + 1) % 100 == 0:
        print("finished {0:.0%}".format(i / adatasub.layers['watershed_labels'].shape[0]))
    for j in range(adatasub.layers['watershed_labels'].shape[1]):
        if (not i % downrs == 0) or (not j % downrs == 0):
            continue
        idx = int(math.floor(i) * math.ceil(patchsizey) + math.floor(j))
        label = adatasub.layers['watershed_labels'][i, j]
        distance=adatasub.layers['watershed_distance'][i,j]
        # For Optimize_train
        if label > 0 and distance >= distance_ratio_threshold:
        # if label > 0 and distance >= distance_ratio_threshold:
            counter_core += 1
            
            dataset_map[i,j]=train_color
            
            x_optimize_sample = [all_exp_merged_bins[idx, :]]
            x_optimize_pos_sample = [[i, j]]
            y_optimize_sample = watershed2center[label]
            
            x_optimize_labels_sample = [distance]
            
            nucleus_center = watershed2center[label]
            for dx, dy in offsets:
                if len(x_optimize_sample) == n_neighbor:
                    break
                x = i + dx
                y = j + dy
                if x < 0 or x >= adatasub.layers['watershed_labels'].shape[0] or y < 0 or y >= \
                                adatasub.layers['watershed_labels'].shape[1]:
                        continue
                idx_nb = int(math.floor(x) * math.ceil(patchsizey) + math.floor(y))
                if 0 <= idx_nb < all_exp_merged_bins.shape[0]:
                    x_optimize_sample.append(all_exp_merged_bins[idx_nb, :])
                    x_optimize_pos_sample.append([x, y])
                    x_optimize_labels_sample.append(adatasub.layers['watershed_distance'][x, y])
            if len(x_optimize_sample) < n_neighbor:
                continue
            x_train.append(x_optimize_sample)
            x_pos.append(x_optimize_pos_sample)
            x_labels.append(np.array(x_optimize_labels_sample))
            y_train.append(np.array(y_optimize_sample))
            y_bin.append(1)
        # For x_test
        elif label == 0:
            counter_outer += 1
            dataset_map[i,j]=train_color
            x_optimize_sample = [all_exp_merged_bins[idx, :]]
            x_optimize_pos_sample = [[i, j]]
            x_optimize_labels_sample = [distance]
            y_optimize_sample = [[-1, -1]]
            for dx, dy in offsets:
                if len(x_optimize_sample) == n_neighbor:
                    break
                x = i + dx
                y = j + dy
                if x < 0 or x >= adatasub.layers['watershed_labels'].shape[0] or y < 0 or y >= \
                    adatasub.layers['watershed_labels'].shape[1]:
                    continue
                idx_nb = int(math.floor(x) * math.ceil(patchsizey) + math.floor(y))
                if 0 <= idx_nb < all_exp_merged_bins.shape[0]:
                    x_optimize_sample.append(all_exp_merged_bins[idx_nb, :])
                    x_optimize_pos_sample.append([x, y])
                    x_optimize_labels_sample.append(adatasub.layers['watershed_distance'][x, y])

            if len(x_optimize_sample) < n_neighbor:
                    continue
            
            x_train_outer.append(x_optimize_sample)
            x_pos_outer.append(x_optimize_pos_sample)
            x_labels_outer.append(np.array(x_optimize_labels_sample))
            y_train_outer.append(np.array([-1,-1]))
            y_bin_outer.append(0)
            
for i in range(adatasub.layers['watershed_labels'].shape[0]):
    if (i + 1) % 100 == 0:
        print("finished {0:.0%}".format(i / adatasub.layers['watershed_labels'].shape[0]))
    for j in range(adatasub.layers['watershed_labels'].shape[1]):
        idx = int(math.floor(i) * math.ceil(patchsizey) + math.floor(j))
        label = adatasub.layers['watershed_labels'][i, j]
        distance = adatasub.layers['watershed_distance'][i, j]
        
        # 忽略已经处理过的核心点和核外点
        # if label > 0 and (distance < distance_ratio_threshold) and (i % downrs == 0) and (j % downrs == 0):
        #     continue
        # if label == 0 and (i % downrs == 0) and (j % downrs == 0):
        #     continue

        
        # dataset_map[i,j]=test_color
        # 对于剩余的点，添加到测试集
        x_test_sample = [all_exp_merged_bins[idx, :]]
        x_test_pos_sample = [[i, j]]
        x_test_labels_sample = [distance]
            
        for dx, dy in offsets:
            if len(x_test_sample) == n_neighbor:
                break
            x = i + dx
            y = j + dy
            if x < 0 or x >= adatasub.layers['watershed_labels'].shape[0] or y < 0 or y >= adatasub.layers['watershed_labels'].shape[1]:
                continue
            idx_nb = int(math.floor(x) * math.ceil(patchsizey) + math.floor(y))
            if 0 <= idx_nb < all_exp_merged_bins.shape[0]:
                x_test_sample.append(all_exp_merged_bins[idx_nb, :])
                x_test_pos_sample.append([x, y])
                x_test_labels_sample.append(adatasub.layers['watershed_distance'][x, y])
            
        if len(x_test_sample) < n_neighbor:
            continue
            
        x_test.append(x_test_sample)
        x_test_pos.append(x_test_pos_sample)
        x_test_labels.append(x_test_labels_sample)


plt.imsave('fig/dataset_map.png', dataset_map)

num_samples = min(len(x_train), len(x_train_outer))

# 随机选择反例数组中的元素
random_indices = np.random.choice(len(x_train_outer), num_samples, replace=False)

x_train = x_train + [x_train_outer[i] for i in random_indices]
x_pos = x_pos + [x_pos_outer[i] for i in random_indices]
x_labels = x_labels + [x_labels_outer[i] for i in random_indices]
y_train = y_train + [y_train_outer[i] for i in random_indices]
y_bin = y_bin + [y_bin_outer[i] for i in random_indices]

x_train = np.array(x_train)
x_pos = np.array(x_pos)
x_labels = np.array(x_labels)
y_train = np.array(y_train)
y_bin = np.array(y_bin)
x_test = np.array(x_test)
x_test_pos = np.array(x_test_pos)
x_test_labels = np.array(x_test_labels)
print(counter_core)
print(counter_outer)
print(x_pos.shape)
print(x_labels.shape)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(x_test_pos.shape)
print(x_test_labels.shape)
# 保存到文件
np.savez_compressed('dataset/x_optimize_train' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz',
                    x_optimize_train=x_train)
np.savez_compressed('dataset/x_optimize_pos' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz',
                    x_optimize_pos=x_pos)
np.savez_compressed('dataset/x_optimize_labels' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz',
                    x_optimize_labels=x_labels)
np.savez_compressed('dataset/y_optimize_train' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz',
                    y_optimize_train=y_train)
np.savez_compressed('dataset/y_bin' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz',
                    y_bin=y_bin)
np.savez_compressed('dataset/x_test' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz', x_test=x_test, x_test_pos=x_test_pos, x_test_labels=x_test_labels)
np.savez_compressed('dataset/x_test_pos' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz', x_test_pos=x_test_pos)
# np.savez_compressed('dataset/x_test_pos' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz', x_test_pos=x_test_pos)
# np.savez_compressed('dataset/x_test_labels' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz', x_test_labels=x_test_labels)
