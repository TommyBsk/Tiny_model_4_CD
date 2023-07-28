import numpy as np
from typing import List
from os.path import join
from matplotlib.image import imread
import h5py as h5
import albumentations as alb
import nrrd


def predict_3_volume(x_volume,y_volume,z_volume):

    # altezza x h x v  asse x
    # profondità x h x w asse y
    # profondità x h x w asse z
    
    x_volume = np.moveaxis(x_volume, 0, -1)
    y_volume = np.moveaxis(y_volume, 0, -1)
    z_volume = np.moveaxis(z_volume, 0, -1)
    
    resize_x = alb.Resize(185, 185)
    resize_y = alb.Resize(185, 218) 
    resize_z = alb.Resize(185, 218)
    
    x_volume = resize_x(image=x_volume)["image"]
    y_volume = resize_y(image=y_volume)["image"]
    z_volume = resize_z(image=z_volume)["image"]
    
    y_volume = np.transpose(y_volume,(0,2,1))
    z_volume = np.transpose(z_volume,(2,0,1))
    

    save_data_to_nrrd(x_volume, "/home/ramat/experiments/exp_tinyCD/exp141/test_model100/x_vol__def.nrrd")
    save_data_to_nrrd(y_volume, "/home/ramat/experiments/exp_tinyCD/exp141/test_model100/y_vol__def.nrrd")
    save_data_to_nrrd(z_volume, "/home/ramat/experiments/exp_tinyCD/exp141/test_model100/z_vol__def.nrrd")
 
    
    volume = x_volume + y_volume + z_volume
    return ((volume/2)>=1).astype(int) 

def load_volume(data_path, volume_axis):
    """
    Load all the masks of the specific axis and compose them in a volume.
    """
    masks_name_list = read_images_list(join(data_path, "mask_name.txt"))
    masks_name_list = [s.strip('\n') for s in masks_name_list]
    masks_list = []
    
    for mask_name in masks_name_list:
         if volume_axis in mask_name:
              mask = imread(join(data_path, mask_name))
              masks_list.append(mask)
              
    return np.stack(masks_list, axis=0)

def read_images_list(txt_file: str) -> List[str]:
        with open(txt_file, "r") as f:
            return f.readlines()
        
def save_data_to_nrrd(data, file_path):
    nrrd.write(file_path, data)
        
def save_data_to_hdf5(data, file_path, internal_path="/data", chunking=True):
    with h5.File(file_path, "w") as f:
        f.create_dataset(
            internal_path, data=data, chunks=chunking
        )


if __name__ == '__main__':
    x_volume = load_volume("/home/ramat/experiments/exp_tinyCD/exp141/test_model100/output_mask", "x")
    y_volume = load_volume("/home/ramat/experiments/exp_tinyCD/exp141/test_model100/output_mask", "y")
    z_volume = load_volume("/home/ramat/experiments/exp_tinyCD/exp141/test_model100/output_mask", "z")
    composed_volume = predict_3_volume(x_volume, y_volume, z_volume)
    #save_data_to_hdf5(composed_volume, "/home/ramat/experiments/exp_tinyCD/exp141/test_model100/3dmodel.h5")
    save_data_to_nrrd(composed_volume, "/home/ramat/experiments/exp_tinyCD/exp141/test_model100/3dmodel_def.nrrd")
