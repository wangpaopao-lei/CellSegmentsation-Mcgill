
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


adatasub.layers['unspliced'] = adatasub.X
patchsizex = adatasub.X.shape[0]
patchsizey = adatasub.X.shape[1]

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
# print(np.min(xall), np.min(yall), np.max(xall), np.max(yall))

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


# 创建 AnnData 对象
all_exp_merged_bins_ad = ad.AnnData(
    all_exp_merged_bins,
    obs=pd.DataFrame(index=[i for i in range(all_exp_merged_bins.shape[0])]),
    var=pd.DataFrame(index=[i for i in range(all_exp_merged_bins.shape[1])]),
)

# 计算高度可变基因
sc.pp.highly_variable_genes(all_exp_merged_bins_ad, n_top_genes=2000, flavor='seurat_v3', span=1.0)
selected_index = all_exp_merged_bins_ad.var[all_exp_merged_bins_ad.var.highly_variable].index
selected_index = list(selected_index)
selected_index = [int(i) for i in selected_index]

all_exp_merged_bins = all_exp_merged_bins.toarray()[:, selected_index]
# 提取选定基因的表达数据
binary_expression = all_exp_merged_bins > 0

# 计算每个基因有表达的像素点个数
pixels_per_gene = binary_expression.sum(axis=0)

# 将结果转换为 numpy 数组
pixels_per_gene = np.array(pixels_per_gene).flatten()  # 如果矩阵很大，这一步可能会占用较多内存

# 绘制直方图
plt.figure(figsize=(8, 6))
plt.hist(pixels_per_gene, bins=100)  # 您可以调整 bins 的数量以更好地显示数据
plt.title("Number of Pixels Expressing Each Gene")
plt.xlabel("Number of Pixels")
plt.ylabel("Frequency of Genes")
plt.savefig('fig/gene_hist.png')

pixels_per_gene = np.array(pixels_per_gene).flatten()

# 使用 numpy 的 histogram 函数来获取 bins 和频率
counts, bin_edges = np.histogram(pixels_per_gene, bins=100)
# 打印数据表
print("Bin_start\tBin_end\tCount")
for i in range(len(counts)):
    if counts[i] > 0:  # 只打印非零计数
        print(f"{bin_edges[i]}\t{bin_edges[i+1]}\t{counts[i]}")


# counts, bin_edges = np.histogram(total_expression_per_gene, bins=100)
# for i in range(len(counts)):
#     print(f"Bin range: {bin_edges[i]} - {bin_edges[i+1]}, Count: {counts[i]}")