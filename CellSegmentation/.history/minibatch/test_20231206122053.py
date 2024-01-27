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
print(adatasub)
fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
st.cs.mask_nuclei_from_stain(adatasub, otsu_classes=4, otsu_index=1)
st.pl.imshow(adatasub, 'stain_mask', ax=ax)
st.cs.find_peaks_from_mask(adatasub, 'stain', 7)
st.cs.watershed(adatasub, 'stain', 5, out_layer='watershed_labels')

fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
st.pl.imshow(adatasub, 'stain', save_show_or_return='return', ax=ax)
# plt.savefig('fig/watershed_labels'+ startx + ':' + starty + ':' + '.png')
st.pl.imshow(adatasub, 'watershed_labels', labels=True, alpha=0.5, ax=ax)
# plt.savefig('fig/watershed_result' + startx + ':' + starty + ':' + '.png')
print(adatasub.layers['watershed_labels'.shape])