import pandas as pd
import os
import numpy as np
import math

def batchlabelcreator(labels):
    for b in labels:
        if b == 1:
            return 1
    return 0


def createbatch(data, datalabel, shift=1800, step=1):
    i = 0
    shape = data.shape
    batchs = []
    batchslabel = []
    nextbatchs=[]
    ycount = 0

    while (i+2) * shift <= shape[0]:
        # 第 i个块
        # print(i,"/",shape[0])
        batch = data[i * shift:(i+1) * shift , :]
        nextbatch=data[(i+1) * shift: (i+2) * shift , :]
        batchs.append(batch)
        nextbatchs.append(nextbatch)
        y = batchlabelcreator(datalabel[(i+1) * shift: (i+2) * shift])
        if y == 1:
            ycount += 1
        batchslabel.append(y)
        i += 1
    print("溢流次数：", ycount)
    return batchs, batchslabel,nextbatchs


def readdata(path,newcolumns):
    files = os.listdir(path)
    j = 0
    totaldata = []

    for f in files:
        frames = []

        df = pd.read_csv(os.path.join(path, f), index_col=0)
        frames.append(df)
        totaldata.append(pd.concat(frames))
        j = j + 1

    alldata = []
    all_next_data=[]
    labels = []
    print("reading_finished")
    print(totaldata[0].describe(include='all'))
    # print(totaldata[1].info(verbose=True,null_counts=True))

    for d in totaldata:

        d = d.fillna(value=0)
        # print(totaldata[1].info(verbose=True,null_counts=True))
        newd = d[newcolumns].values
        shape = newd.shape
        # print(np.isnan(newd).any())
        for j in range(shape[1]):
            mean = np.mean(newd[:, j])
            var = np.var(newd[:, j])
            newd[:, j] = (newd[:, j] - mean) / math.sqrt(var)
        alldata.append(newd)
        # print(np.isnan(newd).any())
        labels.append(d['溢流'].values)
    one_data = []
    one_labels = []
    one_next_data = []
    for i in range(len(alldata)):
        newdata, newlabel,newpre_data = createbatch(alldata[i], labels[i])
        one_data += newdata
        one_next_data+=newpre_data
        one_labels += newlabel


    return one_data, one_labels,one_next_data
