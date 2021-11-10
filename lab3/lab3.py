import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import sounddevice as sd
import soundfile as sf

# data, fs = sf.read("c:\\Users\\mm39362\\Documents\\systemy-multimedialne\\lab3\\sound1.wav", dtype='float32')
# print(data.dtype)
# print(data.shape)

x1 = np.arange(np.iinfo(np.int32).min,np.iinfo(np.int32).max,1000,dtype=np.int32)

x2 = np.round(np.linspace(0,255,255,dtype=np.uint8))
x3 = np.linspace(-1,1,1000)

def change_res(x, bit):
    o_type = x.dtype
    if (o_type == np.float32):
        m = -1
        n = 1
    else:
        m = np.iinfo(o_type).min
        n = np.iinfo(o_type).max
        x = x.astype(np.float32)
    
    rangeC = (x - m)/(n - m)
    d = (2 ** bit) - 1
    rangeD = np.round(rangeC * d) / d

    return ((rangeD * (n - m)) + m).astype(o_type)

# plt.plot(x1, change_res(x1, 2))
# plt.show()

plt.plot(x2, change_res(x2, 2))
plt.show()

plt.plot(x2, change_res(x2, 2))
plt.show()

def downsampling_dec(x, Fs, n):
    return x[::n], round(Fs/n)

from scipy.interpolate import interp1d

def downsampling_interp(x, Fs, nFs, method='lin'):
    if (method == 'lin'):
        metode_lin=interp1d(x,y)


# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(np.arange(0,data.shape[0])/fs,data)

# plt.subplot(2,1,2)
# yf = scipy.fftpack.fft(data)
# plt.plot(np.arange(0,fs,1.0*fs/(yf.size)),np.abs(yf))
# plt.show()

