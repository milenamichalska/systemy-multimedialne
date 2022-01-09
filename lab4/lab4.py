import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# mu law
def encode_mu_law(data, u):
    return np.sign(data) * (np.log(1 + u * np.abs(data)) / np.log(1 + u))

def quantize(data, bit):
    res = data
    start = -1
    end = 1
    q_range = 2 ** bit - 1

    res = (res - start) / (end - start)
    res = np.round(res * q_range) / q_range
    res = ((res * (end - start)) + start)
    return res

def decode_mu_law(data, u):
    return np.sign(data) * (1 / u) * (np.power(1 + u, np.abs(data)) - 1)

def mu_law_compression(data, u, bit):
    res = encode_mu_law(data, u)
    res = quantize(res, bit)
    res = decode_mu_law(res, u)
    return res

# DCPM
def encode_DPCM(data, bit):
    res = data.copy()
    E = res[0]
    for x in range(1, data.shape[0]):
        diff = data[x] - E
        diff = quantize(diff, 16)
        res[x] = diff
        E += diff
    return res

def decode_DPCM(data):
    res = data.copy()
    for x in range(1, data.shape[0]):
        res[x] = res[x - 1] + data[x]
    return res

def DPCM_compression(data, bit):
    res = encode_DPCM(data, bit)
    res = decode_DPCM(res)
    return res

bits = 8
freq, data = wavfile.read('/home/milena/projects/studia/systemy-multimedialne/lab4/sing_low1.wav')

if data.dtype not in [np.float32]:
      data = data.astype(np.float32) / np.iinfo(data.dtype).max

print(freq, data.shape, np.unique(data).size)
if len(data.shape) > 1:
    data = data[:, 0]

data1 = mu_law_compression(data, 255, bits)
data2 = DPCM_compression(data, bits)

max = int(freq*(1/100))
fig, ax = plt.subplots()
ax.plot(np.arange(0, max) / freq, data2[0:max], label="DCMP")
ax.plot(np.arange(0, max) / freq, data1[0:max], label="mu_law")
ax.plot(np.arange(0, max) / freq, data[0:max], label="Original")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
plt.title("Kompresja stratna")
plt.show()
