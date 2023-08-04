import numpy as np
from typing import List
from volume_composition import load_volume
import matplotlib.pyplot as plt


def plot_3d_volume(volume_to_plot:np.ndarray)->None:
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(volume_to_plot)
    plt.savefig("3c_plot_y.png")


# x_volume = load_volume("/home/ramat/experiments/exp_tinyCD/exp141/test_model100/output_mask", "x") #volume_height x h x w
y_volume = load_volume("/home/ramat/experiments/exp_tinyCD/exp141/test_model100/output_mask", "y") #volume_depth x h x w
# z_volume = load_volume("/home/ramat/experiments/exp_tinyCD/exp141/test_model100/output_mask", "z") #volume_depth x h x w
plot_3d_volume(y_volume)