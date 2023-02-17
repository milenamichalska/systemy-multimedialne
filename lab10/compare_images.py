import numpy as np
import cv2
from os import listdir
from os.path import isfile, join

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim

def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

all_imgs = [[], [], []]

mypath = '/home/milena/projects/studia/multimedialne/systemy-multimedialne/lab10'
all_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
methods = ['blur', 'cJPEG', 'noise']
raw_images = [cv2.imread(mypath + '/img2.jpg'), cv2.imread(mypath + '/img1.jpg'), cv2.imread(mypath + '/img3.jpg')]
raw_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in raw_images]

for method in methods:
    imgs = [i for i in all_files if method in i and '.jpg' in i]
    for img in imgs:
        img1 = cv2.imread(mypath + '/' + img)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        if method == 'blur':
            all_imgs[0].append(img1)
        elif method == 'cJPEG':
            all_imgs[1].append(img1)
        else:
            all_imgs[2].append(img1)

mse_matrix = [[], [], []]
for i, container in enumerate(all_imgs):
    for img in container:
        mse_matrix[i].append(mse(raw_images[i], img))

ssim_matrix = [[], [], []]
for i, container in enumerate(all_imgs):
    for img in container:
        ssim_matrix[i].append(ssim(raw_images[i], img))
print(ssim_matrix)
