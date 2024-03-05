from src import scs
import os

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
cell_directory = os.path.dirname(script_directory)
os.chdir(cell_directory)
bin_file = 'data/Mouse_brain_Adult_GEM_bin1_sub.tsv'
image_file = 'data/Mouse_brain_Adult_sub.tif'
scs.segment_cells(bin_file, image_file, align='rigid')
