import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.fftpack

def chromaSubsample(channel, step):
    if step == "4:4:4":
        return channel
    elif step == "4:2:2":
        res = np.zeros((channel.shape[0], int(channel.shape[1]/2)))
        for x in range(0, channel.shape[0]):
            for y in range(0, channel.shape[1], 2):
                res[x][int(y/2)] = channel[x][y]
        return res

def chromaResample(channel, step):
    if step == "4:4:4":
        return channel
    if step == "4:2:2":
        res = np.zeros((channel.shape[0], int(channel.shape[1]*2)))
        for x in range(0, channel.shape[0]):
            for y in range(0, channel.shape[1]):
                res[x][y*2] = channel[x][y]
                res[x][y*2+1] = channel[x][y]
        return res

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a.astype(float), axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.astype(float), axis=0 , norm='ortho'), axis=1 , norm='ortho')

def zigzag(A):
    template= n= np.array([
            [0,  1,  5,  6,  14, 15, 27, 28],
            [2,  4,  7,  13, 16, 26, 29, 42],
            [3,  8,  12, 17, 25, 30, 41, 43],
            [9,  11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
            ])
    if len(A.shape)==1:
        B=np.zeros((8,8))
        for r in range(0,8):
            for c in range(0,8):
                B[r,c]=A[template[r,c]]
    else:
        B=np.zeros((64,))
        for r in range(0,8):
            for c in range(0,8):
                B[template[r,c]]=A[r,c]
    return B

QY= np.array([
        [16, 11, 10, 16, 24,  40,  51,  61],
        [12, 12, 14, 19, 26,  58,  60,  55],
        [14, 13, 16, 24, 40,  57,  69,  56],
        [14, 17, 22, 29, 51,  87,  80,  62],
        [18, 22, 37, 56, 68,  109, 103, 77],
        [24, 36, 55, 64, 81,  104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
        ])

QC= np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        ])

ones = np.ones((8, 8 ))


def compress(channel, samplingStep, Q):
    sampled = chromaSubsample(channel, samplingStep)
    sampled = sampled.astype(int) - 128
    dct = dct2(sampled)

    res = np.zeros(sampled.shape[0]*sampled.shape[1])
    i = 0
    for x in range(0, dct.shape[0], 8):
        for y in range(0, dct.shape[1], 8):
            zz = dct[x:x+8, y:y+8]
            temp = zigzag(zz)
            res[i:i+64] = np.round(temp/Q.flatten()).astype(int)
            i += 64
    return res


def decompress(channel, samplingStep, Q):
    if samplingStep ==  "4:2:2":
        res = np.zeros((int(np.sqrt(channel.shape[0]*2)), int(np.sqrt(channel.shape[0]*2)/2)))
    else:
        res = np.zeros((int(np.sqrt(channel.shape[0])), int(np.sqrt(channel.shape[0]))))

    for idx, i in enumerate(range(0, channel.shape[0], 64)):
        dequantized = channel[i:i+64] * Q.flatten()
        unzigzaged = zigzag(dequantized)

        x = (idx*8) % res.shape[1]
        y = int((idx*8)/res.shape[1])*8
        res[y:y+8, x:x+8] = unzigzaged

    undcted = idct2(res)+128
    undcted = np.clip(undcted, 0, 255).astype(np.uint8)
    resampled = chromaResample(undcted, samplingStep)
    return resampled


def full(channel, samplingStep, Q):
    compressed = compress(channel, samplingStep, Q)
    return decompress(compressed, samplingStep, Q)

# config
sampling = "4:4:4"
one = False

# read and crop image sample
img = cv2.imread("/home/milena/projects/studia/systemy-multimedialne/lab5/img/img3.jpg")

a = 960
b = 960
img = img[a:a+128, b:b+128]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb).astype(int)
Y, Cr, Cb = cv2.split(img2)

Y2 = full(Y, sampling, ones if one else QY)
Cr2 = full(Cr, sampling, ones if one else QC)
Cb2 = full(Cb, sampling, ones if one else QC)

fig, axs = plt.subplots(4, 2 , sharey=True   )
fig.set_size_inches(9,13)
axs[0,0].imshow(img)
axs[1,0].imshow(Y, cmap=plt.cm.gray)
axs[2,0].imshow(Cr, cmap=plt.cm.gray)
axs[3,0].imshow(Cb, cmap=plt.cm.gray)
axs[0,1].imshow(np.dstack([Y2,Cr2,Cb2]).astype(np.uint8))
axs[1,1].imshow(Y2, cmap=plt.cm.gray)
axs[2,1].imshow(Cr2, cmap=plt.cm.gray)
axs[3,1].imshow(Cb2, cmap=plt.cm.gray)

plt.show()