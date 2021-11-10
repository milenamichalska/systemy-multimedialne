import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

img = cv2.imread('pic1.png')
img = np.array(img)

# def getPalette(img):
#     # plt.imshow(img)
#     # plt.show()
    
#     N = 64
#     palette = img / N
#     palette *= N

#     # plt.imshow(palette)
#     # plt.show()
    
#     return np.unique(palette.reshape(-1, img.shape[2]), axis=0)

def colorfit(px, palette):
    return px

# palette = getPalette(img)
# print(img_palette.shape)

palette = np.array([[0. 0. 0.], [0. 0. 1.], [0. 1. 0.], [0. 1. 1.], [1. 0. 0.], [1. 0. 1.], [1. 1. 0.], [1. 1. 1.]])
rand_px = img[random.randint(0, img.shape[0])][random.randint(0, img.shape[0])]

print(colorfit(rand_px, []))
