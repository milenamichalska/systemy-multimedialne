import cv2
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
######   Konfiguracja       ##################################################
##############################################################################

kat='/home/milena/projects/studia/systemy-multimedialne/lab6/'                               # katalog z plikami wideo
plik="clip_4.mp4"                       # nazwa pliku
ile=100                                 # ile klatek odtworzyc? <0 - calosc
key_frame_counter=4                     # co ktora klatka ma byc kluczowa i nie podlegac kompresji
plot_frames=np.array([30,45])           # automatycznie wyrysuj wykresy
auto_pause_frames=np.array([25])        # automatycznie zapauzuj dla klatki i wywietl wykres
subsampling="4:4:4"                     # parametry dla chorma subsamplingu
wyswietlaj_kaltki=True                  # czy program ma wyswietlac kolejene klatki

##############################################################################
####     Kompresja i dekompresja    ##########################################
##############################################################################
class data:
    def init(self):
        self.Y=None
        self.Cb=None
        self.Cr=None


def compress(Y,Cb,Cr, key_frame_Y, key_frame_Cb, key_frame_Cr, samplingStep):
    kdata = data()
    kdata.Y = key_frame_Y
    kdata.Cb = key_frame_Cb
    kdata.Cr = key_frame_Cr

    cdata = data()
    cdata.Y = chromaSubsample(Y, samplingStep)
    cdata.Cb = chromaSubsample(Cb, samplingStep)
    cdata.Cr = chromaSubsample(Cr, samplingStep)
    return cdata, kdata

def decompress(data,  key_frame_Y, key_frame_Cb, key_frame_Cr , samplingStep):
    Y = chromaResample(data.Y, samplingStep)
    Cb = chromaResample(data.Cb, samplingStep)
    Cr = chromaResample(data.Cr, samplingStep)
    frame = np.dstack([Y, Cr, Cb]).astype(np.uint8)
    return frame


##############################################################################
####     Redukcja chrominancji    ##########################################
##############################################################################

def chromaSubsample(channel, step):
    if step == "4:4:4":
        return channel

    if step == "4:2:2":
        res = np.zeros((channel.shape[0], int(channel.shape[1]/2)))
        for x in range(0, channel.shape[0]):
            for y in range(0, channel.shape[1], 2):
                res[x][int(y/2)] = channel[x][y]
        return res

    if step == "4:4:0":
        res = np.zeros((int(channel.shape[0]/2), int(channel.shape[1])))
        print(channel.shape)
        for x in range(0, channel.shape[0], 2):
            for y in range(0, channel.shape[1]):
                res[int(x/2)][y] = channel[x][y]
        return res

    if step == "4:2:0":
        res = np.zeros((int(channel.shape[0]/2), int(channel.shape[1]/2)))
        for x in range(0, channel.shape[0], 2):
            for y in range(0, channel.shape[1], 2):
                res[int(x/2)][int(y/2)] = channel[x][y]
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
    if step == "4:4:0":
        res = np.zeros((channel.shape[0]*2, int(channel.shape[1])))
        for x in range(0, channel.shape[0]):
            for y in range(0, channel.shape[1]):
                res[x*2][y] = channel[x][y]
                res[x*2+1][y] = channel[x][y]
        return res
    if step == "4:2:0":
        res = np.zeros((channel.shape[0]*2, int(channel.shape[1]*2)))
        for x in range(0, channel.shape[0]):
            for y in range(0, channel.shape[1]):
                res[x*2][y*2] = channel[x][y]
                res[x*2+1][y*2+1] = channel[x][y]
                res[x*2+1][y*2] = channel[x][y]
                res[x*2][y*2+1] = channel[x][y]
        return res


##############################################################################
####     Różnica do ramek kluczowych      ##########################################
##############################################################################

def encode_keyframe_diffrence(key, frame):
    for x in range(0, frame.Y.shape[0]):
            for y in range(0, frame.Y.shape[1]):
                frame.Y[x][y] = (frame.Y[x][y] - key.Y[x][y])
                frame.Cr[x][y] = (frame.Cr[x][y] - key.Cr[x][y]) 
                frame.Cb[x][y] = (frame.Cb[x][y] - key.Cb[x][y]) 
    return frame

def decode_keyframe_diffrence(key, frame):
    for x in range(0, frame.Y.shape[0]):
            for y in range(0, frame.Y.shape[1]):
                frame.Y[x][y] = (frame.Y[x][y] + key.Y[x][y]) 
                frame.Cr[x][y] = (frame.Cr[x][y] + key.Cr[x][y]) 
                frame.Cb[x][y] = (frame.Cb[x][y] + key.Cb[x][y]) 
    return frame


##############################################################################
####     Głowna petla programu      ##########################################
##############################################################################

cap = cv2.VideoCapture(kat+plik)

if ile<0:
    ile=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cv2.namedWindow('Normal Frame')
cv2.namedWindow('Decompressed Frame')

compression_information=np.zeros((3,ile))

for i in range(ile):
    ret, frame = cap.read()
    if wyswietlaj_kaltki:
        cv2.imshow('Normal Frame',frame)
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    if (i % key_frame_counter)==0: # pobieranie klatek kluczowych
        key_frame=frame
        cY=frame[:,:,0]
        cCb=frame[:,:,2]
        cCr=frame[:,:,1]
        d_frame=frame
    else: # kompresja
        cdata, key_data =compress(frame[:,:,0],frame[:,:,2],frame[:,:,1], key_frame[:,:,0], key_frame[:,:,2], key_frame[:,:,1], samplingStep=subsampling)
        cdata = encode_keyframe_diffrence(key_data, cdata)
        cdata = decode_keyframe_diffrence(key_data, cdata)

        cY=cdata.Y
        cCb=cdata.Cb
        cCr=cdata.Cr
        d_frame= decompress(cdata, key_frame[:,:,0], key_frame[:,:,2], key_frame[:,:,1], samplingStep=subsampling)
    
    compression_information[0,i]= (frame[:,:,0].size - cY.size)/frame[:,:,0].size
    compression_information[1,i]= (frame[:,:,0].size - cCb.size)/frame[:,:,0].size
    compression_information[2,i]= (frame[:,:,0].size - cCr.size)/frame[:,:,0].size  
    if wyswietlaj_kaltki:
        cv2.imshow('Decompressed Frame',cv2.cvtColor(d_frame,cv2.COLOR_YCrCb2BGR))
    
    if np.any(plot_frames==i): # rysuj wykresy
        # bardzo słaby i sztuczny przyklad wykrozystania tej opcji
        fig, axs = plt.subplots(1, 3 , sharey=True   )
        fig.set_size_inches(16,5)
        axs[0].imshow(frame)
        axs[2].imshow(d_frame) 
        diff=frame.astype(float)-d_frame.astype(float)
        print(np.min(diff),np.max(diff))
        axs[1].imshow(diff,vmin=np.min(diff),vmax=np.max(diff))
        
    if np.any(auto_pause_frames==i):
        cv2.waitKey(-1) #wait until any key is pressed
    
    k = cv2.waitKey(1) & 0xff
    
    if k==ord('q'):
        break
    elif k == ord('p'):
        cv2.waitKey(-1) #wait until any key is pressed

plt.figure()
plt.plot(np.arange(0,ile),compression_information[0,:]*100)
plt.plot(np.arange(0,ile),compression_information[1,:]*100)
plt.plot(np.arange(0,ile),compression_information[2,:]*100)