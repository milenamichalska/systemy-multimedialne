# https://mkramarczyk.zut.edu.pl/?cat=D&l=02

import numpy as np
from skimage import io
from skimage.color import rgb2lab ,lab2lch

brightness = ''
temp = ''
color_contrast = ''
contrast = ''
saturation = ''

mypath = '/home/milena/projects/studia/multimedialne/systemy-multimedialne/lab10/img1.jpg'
img = io.imread(mypath)

lab_img = rgb2lab(img)
lch_img = lab2lch(lab_img)

b = np.mean(lab_img[:,:,0] * (100.0/lab_img[:,:,0].max()))

if (b < 25):
    brightness = 'bardzo ciemne'
if (b > 25 and b < 42):
    brightness = 'ciemne'
if (b >= 42 and b < 58):
    brightness = 'średnie'
if (b >= 58 and b < 75):
    brightness = 'jasne'
if (b >= 75):
    brightness = 'bardzo'

h = np.mean(np.rad2deg(lch_img[:,:,2]))
if (h < 100 or h > 340):
    temp = 'ciepłe'
elif (h >160 and h < 300):
    temp = 'zimne'
else:
    temp = 'neutralne'

c = np.mean(lch_img[:,:,1] * (100.0/lch_img[:,:,1].max()))
if (c < 18.5):
    saturation = 'niską'
if (c >=18.5 and c < 39):
    saturation = 'średnią'
if (c >= 39):
    saturation = 'wysoką'

L_s = b
a_s = np.mean(lab_img[:, :, 1])
b_s = np.mean(lab_img[:, :, 2])

flat_img = lab_img.reshape((lab_img.shape[0] * lab_img.shape[1] ,3))

sum_c = 0
sum_color_c = 0

for pixel in flat_img:
    sum_c += (pixel[0] - L_s)**2 / flat_img.shape[0]
    sum_color_c += ((pixel[1] - a_s)**2 + (pixel[2] - b_s)**2) / flat_img.shape[0]

contrast_value = np.sqrt(sum_c)
color_contrast_value = np.sqrt(sum_color_c)

print(contrast_value)
print(color_contrast_value)

if (contrast_value >= 30):
    contrast = 'wysokim'
elif (contrast_value < 30 and contrast_value >= 10):
    contrast = 'średnim'
else:
    contrast = 'małym'

if (color_contrast_value >= 25):
    color_contrast = 'wysokim'
elif (contrast_value < 25 and contrast_value >= 15):
    color_contrast = 'średnim'
else:
    color_contrast = 'małym'

print('W obrazie dominują {} {} kolory. Cechuje się on {} kontrastem barwnym, {} kontrastem oraz {} saturacją.'.format(brightness, temp, color_contrast, contrast, saturation))