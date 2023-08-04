import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
import random


def find_deformation_point(img):
    laplacian_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    filtered_img = cv.filter2D(src=img, kernel=laplacian_kernel, ddepth=-1) 
    borderPoints_list = []
    for row in range(0,img.shape[0]):
        for column in range(0,img.shape[1]):
            if filtered_img[row][column]!=0:
                borderPoints_list.append((column,row))
    return random.choice(borderPoints_list)

def apply_deformation(img, point, radius=45, power=1.6):
    height, width = img.shape
    map_y = np.zeros((height,width),dtype=np.float32)
    map_x = np.zeros((height,width),dtype=np.float32)

    # create index map
    for i in range(height):
        for j in range(width):
            map_y[i][j]=i
            map_x[i][j]=j

    # deform around the right eye
    for i in range (-radius, radius):
        for j in range(-radius, radius):
            if (i**2 + j**2 > radius ** 2) or i < 0 or j < 0:
                continue

            if i > 0:
                map_y[point[1] + i][point[0] + j] = point[1] + (i/radius)**power * radius
            if i < 0:
                map_y[point[1] + i][point[0] + j] = point[1] - (-i/radius)**power * radius
            if j > 0:
                map_x[point[1] + i][point[0] + j] = point[0] + (j/radius)**power * radius
            if j < 0:
                map_x[point[1] + i][point[0] + j] = point[0] - (-j/radius)**power * radius

    warped=cv.remap(img,map_x,map_y,cv.INTER_LINEAR)
    cv.imwrite('result_img.jpg', warped)
    
if __name__=="__main__":
    img_ = cv.imread("C:/Users/tommy/Downloads/data0_z_stack_94.png")
    gray_image = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)
    p = find_deformation_point(gray_image)
    apply_deformation(gray_image, p)