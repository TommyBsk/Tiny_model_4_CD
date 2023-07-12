import numpy as np

def predict_3_volume(x_volume,y_volume,z_volume):
    volume = x_volume +y_volume+ z_volume
    return ((volume/2)>=1).astype(int) 

def load_volume(data_path, volume_axis):
    """
    Load all the masks of the specific axis and compose them in a volume.
    """
    pass


if __name__ == '__main__':
    # x_volume = load_volume(data_path, "X")
    # y_volume = load_volume(data_path, "Y")
    # z_volume = load_volume(data_path, "Z")
    # composed_volume = predict_volume(...)
    # save_to_hdf5/visualize_volume
    pass