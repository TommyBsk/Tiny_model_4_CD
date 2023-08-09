import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
import random


def find_deformation_point(img):
    laplacian_kernel = np.array([[-1,-1,-1],
                                    [-1,8,-1],
                                    [-1,-1,-1]], np.float32)
    filtered_img = cv.filter2D(src=np.float32(img), kernel=laplacian_kernel, ddepth=-1) 
    borderPoints_list = []
    for row in range(0,img.shape[0]):
        for column in range(0,img.shape[1]):
            if filtered_img[row][column]!=0:
                borderPoints_list.append((column,row))
    if borderPoints_list:
        return random.choice(borderPoints_list)
    else:
        return None

def deform_image(src_img, src_mask, point, radius_lower_boud=25, radius_upper_bound=45, power_upper_bound=1.6):
    img = src_img.copy()
    mask = src_mask.copy()
    
    radius = int(random.uniform(radius_lower_boud,radius_upper_bound))
    power = random.uniform(0,power_upper_bound)

    if len(img.shape) > 2:
        height, width, _ = img.shape
    else:
        height, width = img.shape
        
    map_y = np.zeros((height,width),dtype=np.float32)
    map_x = np.zeros((height,width),dtype=np.float32)

    # create index map
    for i in range(height):
        for j in range(width):
            map_y[i][j]=i
            map_x[i][j]=j

    # deform around the deformation point
    for i in range (-radius, radius):
        for j in range(-radius, radius):
            if (i**2 + j**2 > radius ** 2):
                continue

            if i > 0:
                map_y[clippig_bound((point[1] + i),height)][clippig_bound((point[0] + j),width)] = point[1] + (i/radius)**power * radius
            if i < 0:
                map_y[clippig_bound(point[1] + i,height)][clippig_bound(point[0] + j, width)] = point[1] - (-i/radius)**power * radius
            if j > 0:
                map_x[clippig_bound(point[1] + i, height)][clippig_bound(point[0] + j, width)] = point[0] + (j/radius)**power * radius
            if j < 0:
                map_x[clippig_bound((point[1] + i),height)][clippig_bound((point[0] + j),width)] = point[0] - (-j/radius)**power * radius

    deformed_image = cv.remap(img,map_x,map_y,cv.INTER_NEAREST)
    deformed_mask = cv.remap(mask,map_x,map_y,cv.INTER_NEAREST)

    return deformed_image, deformed_mask

def apply_deformation(image, mask):
    deformation_point = find_deformation_point(mask)
    if deformation_point is None:
        return image, mask

    return deform_image(src_img=image, src_mask=mask, point=deformation_point)

def clippig_bound(value_to_clip,upper_bound):
    return np.clip(value_to_clip,0,upper_bound)


    