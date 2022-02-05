import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

path = '/home/milena/projects/studia/systemy-multimedialne/lab10/results.csv'
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

img3_cols = ['gauss3).', 'gauss10',
       'gauss12', 'gauss15', 'gauss20', 'rand5', 'rand25', 'rand37', 'rand50']

img_cols = [img1_cols, img2_cols, img3_cols]

def draw_plots(df):
    for cols in img_cols:
        cols.append('id')
        data1 = df[cols]
        for x, col in enumerate(data1.columns):
            if col == 'id':
                continue
            for y, ind in enumerate(data1.index):
                if data1.loc[ind, col]:
                    plt.plot(col, data[col][ind], 'o', color=colors[data['id'][ind]])
        
        # d = np.polyfit(data1.columns, data1.index,1)
        # f = np.poly1d(d)
        # data1.insert(6,'Treg',f(data1.columns))
        # data1.plot(x=data1.columns, y='Treg',color='red',ax=ax)

        plt.xticks(range(len(data1.columns)), labels=data1.columns)
        plt.show()

#MOS dla obrazów - wszystkie
draw_plots(data)

#MOS dla obrazów - zagregowane dla badanego
data_agg_by_id = data.groupby('id').mean().reset_index()
draw_plots(data_agg_by_id)

#MOS dla obrazów - zagregowane
data_agg = data.mean().reset_index()

for i in range(3):
    data1 = data_agg[i*9:((i+1)*9) - 1].reset_index(drop=True)
    data1.plot(style='.')
    plt.xticks(range(8), labels=data1['index'])
    plt.show()

def mse(org, modified):
    m, n, _ = org.shape
    sum = 0

    for k in range(1):
        for i in range(m-1):
            for j in range(n-1):
                sum += abs(org[i,j,k] - modified[i,j,k])**2

    return (((1/(m*n))*sum))

def psnr(org, modified, max):
    return 10 * math.log10(((max)**2)/(mse(modified, org)))

