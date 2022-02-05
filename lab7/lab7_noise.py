import cv2
import matplotlib.pyplot as plt
import numpy as np

path = '/home/milena/projects/studia/systemy-multimedialne/lab8/'
file_name = 'img3.jpg'
q_factor = 37

img = cv2.imread(path + file_name)

# gauss = np.random.normal(0,q_factor,(img.shape))
# noisy = (img + gauss).clip(0,255).astype(np.uint8)

rand = q_factor*np.random.random((img.shape))
noisy = (img + rand).clip(0,255).astype(np.uint8)

cv2.imwrite('noise_rand' + str(q_factor) + file_name, noisy)

