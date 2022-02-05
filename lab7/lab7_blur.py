import cv2
import matplotlib.pyplot as plt

path = '/home/milena/projects/studia/systemy-multimedialne/lab8/'
file_name = 'img2.jpg'
q_factor = 7

img = cv2.imread(path + file_name)

# blur = cv.blur(img,(5,5))
# blur = cv2.GaussianBlur(img,(q_factor,q_factor),0)
# median = cv2.medianBlur(img,q_factor)
blur = cv2.bilateralFilter(img,q_factor,75,75)

cv2.imwrite('blur_bilateral' + str(q_factor) + file_name, blur)

