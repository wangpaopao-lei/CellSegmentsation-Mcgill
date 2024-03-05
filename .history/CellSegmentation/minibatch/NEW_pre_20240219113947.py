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


def read_data(bin_file, image_file):
    """
    read transcripts data and image data
    """
    return st.io.read_bgi_agg(bin_file, image_file)

def crop_image_data(adatasub, start_x, start_y, end_x, end_y):
    """
    crop a smaller dataset for validation
    """
    adatasub.layers['unspliced'] = adatasub.X
    patchsizex = adatasub.X.shape[0]
    patchsizey = adatasub.X.shape[1]
    
    cropped_image_data = adatasub.layers['stain'][start_y:end_y, start_x:end_x]
    
    empty_X=np.zeros((cropped_image_data))
    adatasub=ad.AnnData(X=empty_X,uns=adatasub.uns.copy())
    adatasub.layers['stain']=cropped_image_data
    return adatasub

def segment_nucleus(adatasub):
    """
    segment nucleus and save mask images
    """
    st.cs.mask_nuclei_from_stain(adatasub, otsu_classes=4, otsu_index=1)
    st.cs.find_peaks_from_mask(adatasub, 'stain', 7)
    st.cs.watershed(adatasub, 'stain', 5, out_layer='watershed_labels')
    
    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
    st.pl.imshow(adatasub, 'watershed_labels', labels=True, alpha=0.5, ax=ax)
    plt.savefig('fig/watershed_result.png')
    
    with open('dataset/watershed_labels.txt','w') as fw:
        for i in range(adatasub.layers['watershed_labels'].shape[0]):
            for j in range(adatasub.layers['watershed_labels'].shape[1]):
                fw.write(str(i) + '\t' + str(j) + '\t' + str(adatasub.layers['watershed_labels'][i][j]) + '\n')
        
    return adatasub
    
    
def calculate_distance_map(adatasub):
    """
    calculate and add watershed_distance layer
    """
    labels = adatasub.layers['watershed_labels']
    unique_labels = np.unique(labels)[1:]  # 移除背景标签
    probability_map = np.zeros_like(labels, dtype=float)
    
    for label in unique_labels:
        binary_cell = (labels == label)
        boundaries = segmentation.find_boundaries(binary_cell, mode='inner')
        distance = distance_transform_edt(np.where(binary_cell, 1-boundaries, 0))
        max_distance = distance.max()
        probability_map[labels == label] = distance / max_distance if max_distance else 0.5
    adatasub.layers['watershed_distance'] = probability_map
    
    return adatasub

def preproccess_gene_data(adatasub,bin_file):
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
    
    


    
    
    