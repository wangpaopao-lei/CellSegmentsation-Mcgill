#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/17 18:26
# @Author  : WangLei
# @File    : validate.py
# @Description :


import numpy as np
import anndata as ad

def calculate_correlation(area1_gene_expressions, area2_gene_expressions):
    """计算两个区域平均基因表达向量的相关性."""
    # 计算每个区域的平均基因表达向量
    avg_area1 = np.mean(area1_gene_expressions, axis=0)
    avg_area2 = np.mean(area2_gene_expressions, axis=0)
    return np.corrcoef(avg_area1, avg_area2)[0, 1]

def compare_segmentation_results(result1, result2, gene_expression):
    """比较两种细胞核分割方法的性能."""
    # 寻找两种结果的交集和差集
    intersection = np.logical_and(result1, result2)
    difference1 = np.logical_and(result1, np.logical_not(result2))
    difference2 = np.logical_and(result2, np.logical_not(result1))
    
    # 计算不同区域的基因表达向量
    area1_gene_expressions = gene_expression[intersection]
    area2_gene_expressions = gene_expression[difference1]
    area3_gene_expressions = gene_expression[difference2]
    
    # 计算基因表达向量的相关性
    corr_area1_area2 = calculate_correlation(area1_gene_expressions, area2_gene_expressions)
    corr_area1_area3 = calculate_correlation(area1_gene_expressions, area3_gene_expressions)
    
    return corr_area1_area2, corr_area1_area3




filename='results/spot_prediction_0:0:0:0.txt'
adatasub=ad.read_h5ad('dataset/spots0:0::0:0.h5ad')
wl=adatasub.layers['watershed_labels']
count1=0
count2=0
count3=0
count4=0
with open(filename) as fr:
    for line in fr:
        count3+=1
        
        x,y,b=line.split()

        # 确定点归属
        # 
        if float(b)>0.9:
            count1+=1
        elif float(b)<0.1:
            count4+=1
            
for x in range(wl.shape[0]):
    for y in range(wl.shape[1]):
        if wl[x,y]>0:
            count2+=1
print(count1)
print(count2)
print(count3)
print(count4)
# result1 = np.array([[...]])  # 你的Watershed算法的结果
# result2 = np.array([[...]])  # 你自己改进的算法的结果
# gene_expression = np.array([[[...]]])  # 基因表达向量，现在它是一个三维数组

# # 使用上述函数
# corr_area1_area2, corr_area1_area3 = compare_segmentation_results(result1, result2, gene_expression)
# print(f"区域1和区域2的平均基因表达向量的相关性: {corr_area1_area2}")
# print(f"区域1和区域3的平均基因表达向量的相关性: {corr_area1_area3}")
