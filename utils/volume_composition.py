import numpy as np
from typing import List
from os.path import join
from matplotlib.image import imread
import h5py as h5

def predict_3_volume(x_volume,y_volume,z_volume):
    breakpoint()
    volume = x_volume + y_volume + z_volume
    return ((volume/2)>=1).astype(int) 

def load_volume(data_path, volume_axis):
    """
    Load all the masks of the specific axis and compose them in a volume.
    """
    masks_name_list = read_images_list(join(data_path, "mask_name.txt"))
    masks_name_list = [s.strip('\n') for s in masks_name_list]
    masks_list = np.empty(0)
    
    for mask_name in masks_name_list:
         if volume_axis in mask_name:
              mask = imread(join(data_path, mask_name))
              masks_list = np.append(masks_list, mask)
              
    return masks_list

def read_images_list(txt_file: str) -> List[str]:
        with open(txt_file, "r") as f:
            return f.readlines()
        
def save_data_to_hdf5(data, file_path, internal_path="/data", chunking=True):
    with h5.File(file_path, "w") as f:
        f.create_dataset(
            internal_path, data=data, chunks=chunking
        )


if __name__ == '__main__':
    x_volume = load_volume("/home/ramat/experiments/exp_tinyCD/exp139/test_model37/output_mask", "x")
    y_volume = load_volume("/home/ramat/experiments/exp_tinyCD/exp139/test_model37/output_mask", "y")
    z_volume = load_volume("/home/ramat/experiments/exp_tinyCD/exp139/test_model37/output_mask", "z")
    composed_volume = predict_3_volume(x_volume, y_volume, z_volume)
    save_data_to_hdf5(composed_volume, "/home/ramat/experiments/exp_tinyCD/exp139/test_model37/3dmodel.h5")
