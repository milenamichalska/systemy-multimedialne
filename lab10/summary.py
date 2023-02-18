import matplotlib

from compare_images import PSNR_matrix
matplotlib.use('tkagg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize

path = '/home/milena/projects/studia/multimedialne/systemy-multimedialne/lab10/results.csv'
data = pd.read_csv(path)

rename_dict = {'Słowo identyfikujące' : 'id', 'Sygnatura czasowa': 'time'}

for column in data.columns:
    if 'Oceń' in column:
        rename_dict[column] = column[13:-1]

data = data.rename(columns=rename_dict).drop(columns=['time'])

colors = {
    'milena': 'yellow',
    'Gosia': 'orange',
    'Henio': 'pink',
    'Pawel': 'red',
    'Robert': 'green',
    'Soltys ': 'blue',
    'broxi': 'brown',
    'kotanakin': 'gray',
    'wanna': 'black'
}

img1_cols = ['jpeg3', 'jpeg5', 'jpeg10', 'jpeg15', 'jpeg20', 'jpeg25',
       'jpeg30', 'jpeg40', 'jpeg80']

img2_cols = ['bi5', 'bi7', 'gauss1', 'gauss3',
       'gauss5', 'gauss7', 'med1', 'med3', 'med5']

img3_cols = ['gauss3', 'gauss10',
       'gauss12', 'gauss15', 'gauss20', 'rand5', 'rand25', 'rand37', 'rand50']

img_cols = [img1_cols, img2_cols, img3_cols]

def draw_plots(df):
    for cols in img_cols:
        cols.append('id')
        data1 = df[cols]
        regression_data = []

        # model = LinearRegression()
        # model.fit(df[cols],[5, 5, 4 ])

        for x, col in enumerate(data1.columns):
            if col == 'id':
                continue
            for y, ind in enumerate(data1.index):
                if data1.loc[ind, col]:
                    plt.plot(col, data[col][ind], 'o', color=colors[data['id'][ind]])
                    regression_data.append(data[col][ind])

        # ids = [*np.arange(0, len(regression_data)/9, 1/9)]
        ids = [*np.arange(0, len(regression_data)/30, 1/30)]
        print(len(regression_data))
        print(len(ids))
        m, b = np.polyfit(ids, regression_data, 1)
        plt.plot(ids, [float(x) * m + b for x in ids])
        
        # d = np.polyfit(data1.columns, data1.index,1)
        # f = np.poly1d(d)
        # data1.insert(6,'Treg',f(data1.columns))
        # data1.plot(x=data1.columns, y='Treg',color='red',ax=ax)

        plt.xticks(range(len(data1.columns)), labels=data1.columns)
        plt.show()

#MOS dla obrazów - wszystkie
# draw_plots(data)

#MOS dla obrazów - zagregowane dla badanego
data_agg_by_id = data.groupby('id').mean().reset_index()
# draw_plots(data_agg_by_id)

#MOS dla obrazów - zagregowane
data_agg = data.mean().reset_index()

# for i in range(3):
#     data1 = data_agg[i*9:((i+1)*9) - 1].reset_index(drop=True)
#     data1.plot(style='.')
#     ids = [*np.arange(0, len(data1), 1)]
#     m, b = np.polyfit(ids, data1[0], 1)
#     plt.plot(ids, [float(x) * m + b for x in ids])

#     plt.xticks(range(8), labels=data1['index'])
#     plt.show()

mse_matrix = np.array([[18.563525390625, 21.153955078125, 16.18890625, 11.9638671875, 15.16677734375, 0.139375, 9.834619140625, 11.53484375, 0.139375], [17.53703180212014, 14.082544169611307, 13.47523851590106, 12.399363957597172, 48.38682862190813, 16.88310070671378, 7.45958480565371, 15.25625441696113, 21.729310954063603], [32.103341121495326, 0.012978971962616822, 27.167266355140185, 22.767628504672896, 0.0060630841121495326, 19.245280373831775, 0.0029789719626168226, 12.31142523364486, 0.04242990654205608]])
ssim_matrix = np.array([[0.8495328732653618, 0.7822021598943217, 0.9098509903437052, 0.8891635198313441, 0.7681451812313564, 0.9990985938224084, 0.9652768508201988, 0.9534778816524874, 0.9990985938224084], [0.8306170431198528, 0.9069697492493021, 0.9145081915535832, 0.9317680990360756, 0.43171848101118726, 0.846677855448326, 0.9722432671561904, 0.8861529381077032, 0.7687577961911227], [0.5954902141350337, 0.8037091466526205, 0.6782659060097556, 0.740728198143139, 0.6900389020338416, 0.7861961126830832, 0.5861747867278111, 0.8618775561803703, 0.9912518863392479]])
psnr_matrix = np.array([[35.444199044859, 34.876887831037976, 36.03862852796158, 37.352087776433706, 36.32187049725766, 56.68895480475675, 38.20322814947855, 37.51068644886602, 56.68895480475675], [35.69124271327285, 36.6439923880264, 36.83543899794075, 37.19680952866197, 31.283532024323666, 35.85628149849016, 39.40365705195718, 36.29632438370534, 34.76034405996652], [33.06530127368991, 66.99840066604196, 33.79034420153594, 34.55762564375368, 70.30386767696606, 35.287561183954175, 73.39013945111108, 37.22772028872551, 61.85408285696016]])
if_matrix = np.array([[0.9460125147573405, 0.9276283076212019, 0.9647869665918798, 0.9517177060713256, 0.9068109021870374, 0.9998968237267195, 0.9921668688551848, 0.9896178540526659, 0.9998968237267195], [0.9155225636925262, 0.9525714423778894, 0.9578116268457989, 0.9662886531457979, 0.7970027993524459, 0.9342357033202396, 0.9884995576531925, 0.9449164118508768, 0.8707529480485237], [0.9401554919771875, 0.9430019527862536, 0.9643101369520275, 0.9761848741988464, 0.8817034173165452, 0.9824957580237503, 0.7967866938125779, 0.9906017094735382, 0.9980085295618777]])

measures = [mse_matrix, psnr_matrix, if_matrix, ssim_matrix]

for i in range(4):
    for measure in measures:
        data1 = measure[i]
        ids = [*np.arange(0, len(data1), 1)]
        plt.scatter(ids, data1)
        m, b = np.polyfit(ids, data1, 1)
        plt.plot(ids, [float(x) * m + b for x in ids])
        plt.show()

mos_matrix = np.reshape(np.array(data_agg[0]), mse_matrix.shape)

mse_matrix = normalize(mse_matrix, axis=1, norm='l1').flatten()
ssim_matrix = normalize(ssim_matrix, axis=1, norm='l1').flatten()
mos_matrix = normalize(mos_matrix, axis=1, norm='l1').flatten()

xyz1 = pd.DataFrame({'mos': mos_matrix[:9], 'mse': mse_matrix[:9], 'ssim': ssim_matrix[:9]})
xyz2 = pd.DataFrame({'mos': mos_matrix[9:18], 'mse': mse_matrix[9:18], 'ssim': ssim_matrix[9:18]})
xyz3 = pd.DataFrame({'mos': mos_matrix[18:], 'mse': mse_matrix[18:], 'ssim': ssim_matrix[18:]})
# con_matrix = np.array([[mos_matrix.flatten()[i], mse_matrix.flatten()[i], ssim_matrix.flatten()[i]] for i in range(len(mos_matrix.flatten()))])
# print(con_matrix.shape)

# for xyz in [xyz1, xyz2, xyz3]:
#     corr_matrix = np.corrcoef(xyz).round(decimals=2)

#     fig, ax = plt.subplots()
#     im = ax.imshow(corr_matrix)
#     ax.xaxis.set(ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8), ticklabels=('MOS1', 'MSE1', 'SSIM1', 'MOS2', 'MSE2', 'SSIM2', 'MOS3', 'MSE3', 'SSIM3'))
#     ax.yaxis.set(ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8), ticklabels=('MOS1', 'MSE1', 'SSIM1', 'MOS2', 'MSE2', 'SSIM2', 'MOS3', 'MSE3', 'SSIM3'))
#     for i in range(corr_matrix.shape[0]):
#         for j in range(corr_matrix.shape[1]):
#             ax.text(j, i, corr_matrix[i, j], ha='center', va='center',
#                     color='r')
#     cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
#     plt.show()