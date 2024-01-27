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