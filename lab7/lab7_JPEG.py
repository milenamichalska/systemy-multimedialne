import cv2
import matplotlib.pyplot as plt

path = '/home/milena/projects/studia/systemy-multimedialne/lab8/'
file_name = 'img1.jpg'
q_factor = 25

img = cv2.imread(path + file_name)

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q_factor]
result, encimg = cv2.imencode('.jpg', img, encode_param)
decimg = cv2.imdecode(encimg, 1)

fig, axs = plt.subplots(1, 2 , sharey=True   )
axs[0].imshow(img)
axs[1].imshow(decimg)

cv2.imwrite('cJPEG' + str(q_factor) + file_name, decimg)

