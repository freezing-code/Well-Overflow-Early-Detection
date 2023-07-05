import datetime
import os.path
import utils
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from cnnlstm import CNNLSTM
from pipeline import  masknet_train
from mymodels import MTSTco
import tool


import json

newcolumns = ['扭矩(KN.m)', '泵冲2(spm)', '立压log(MPa)', '大钩负荷(KN)', '大钩位置(m)',
              '平均钻速(m/min)', '平均钻压(KN)', '入口流量log(L/s)', '出口流量log(%)', '总池体积(m3)', '迟到时间(min)', '定点压耗(MPa)',
              '迟到时间(min)', '入口温度(℃)', '出口温度(℃)', '总烃(%)', 'PWD垂深(m)', 'PWD环空压力(MPa)', 'PWD井斜(deg)', 'PWD方位(deg)',
              'C2(%)', '目标回压(MPa)']


def run(params, dataset, savepath,newcolumns):
    data_x,data_y,data_next_x=utils.readdata(dataset,newcolumns)
    train = np.array(data_x)
    next_train = np.array(data_next_x)

    params['feature_dim'] = train.shape[2]
    params['time_dim'] = train.shape[1]
    params['classnums'] = max(data_y) + 1
    train = train.transpose(0, 2, 1)
    next_train = next_train.transpose(0, 2, 1)

    model = CNNLSTM(xcol_size=params['feature_dim'])
    factory = masknet_train(epoch=1, params=params, model=model, lr=0.01, cuda=True, gpu=0,
                            classifier=None, earlystopping=40)
    # pretain = factory.mask_pretain(train, next_train, accumulation_steps=1, verbose=True, savepath=savepath)

    train, test, train_labels, test_labels = train_test_split(np.array(data_x), np.array(data_y),
                                                              stratify=np.array(data_y))
    print("trainsize:", train.shape)
    print("testsize:", test.shape)


    params['classnums'] = max(train_labels) + 1
    train = train.transpose(0, 2, 1)
    test = test.transpose(0, 2, 1)
    torch.cuda.empty_cache()
    # summary(pretain,(train.shape[2],train.shape[1]))


    positive_ratio = np.sum(train_labels) / train_labels.shape[0]
    negative_ratio = 1 - positive_ratio
    print("weight:", 1 / positive_ratio, 1 / negative_ratio)
    sample_weighted_list = []
    for label in train_labels:
        if label == 1:
            sample_weighted_list.append(1 / positive_ratio)
        else:
            sample_weighted_list.append(1 / negative_ratio)

    #disable sample_weight
    sample_weighted_list =np.ones(train_labels.shape[0])

    deepf, acc = factory.run_baselines(model, train, train_labels, test, test_labels, epoch=400,
                                          lr=0.01, verbose=True,early_stop=50, savepath=savepath,samples_weight=sample_weighted_list)
    df = pd.DataFrame.from_dict(params, orient='index').transpose()
    df.to_csv(os.path.join(savepath, 'params.csv'))

    os.rename(savepath, savepath + str(acc))

def run_mlmodels(params, dataset, savepath,newcolumns):
    tool.mkdir(savepath)
    data_x, data_y, data_next_x = utils.readdata(dataset, newcolumns)
    data=np.array(data_x)
    data=data.reshape((data.shape[0],data.shape[1] * data.shape[2]))
    # pca
    pca = PCA(n_components=0.95, svd_solver='full')
    pca.fit(data)
    data = pca.fit_transform(data)

    train, test, train_labels, test_labels = train_test_split(data, np.array(data_y),
                                                              stratify=np.array(data_y))
    print("trainsize:", train.shape)
    print("testsize:", test.shape)

    params['classnums'] = max(train_labels) + 1


    model = svm.SVC(C=3, cache_size=200, class_weight='balanced'  # , coef0=0.0
                    , decision_function_shape='ovr', degree=3, gamma='auto'
                    , kernel='rbf', max_iter=-1, probability=False, random_state=None
                    )
    model.fit(train, train_labels)

    result = model.predict(test)
    acc2 = metrics.accuracy_score(test_labels, result)
    print("SVM val_acc:", acc2)
    s = metrics.classification_report(test_labels, result, output_dict=True)
    print(metrics.classification_report(test_labels, result))
    df = pd.DataFrame.from_dict(dict(s)).transpose()
    df.to_csv(os.path.join(savepath, 'svm_result.csv'))

    model = KNeighborsClassifier(n_neighbors=5)


    model.fit(train, train_labels)


    result = model.predict(test)
    acc3 = metrics.accuracy_score(test_labels, result)
    print("KNN val_acc:", acc3)
    s = metrics.classification_report(test_labels, result, output_dict=True)
    print(metrics.classification_report(test_labels, result))
    df = pd.DataFrame.from_dict(dict(s)).transpose()
    df.to_csv(os.path.join(savepath, 'knn_result.csv'))

def run_lgb(params, dataset, savepath,newcolumns):
    tool.mkdir(savepath+"lgb")
    tool.mkdir(savepath)
    data_x, data_y, data_next_x = utils.readdata(dataset, newcolumns)
    data=np.array(data_x)
    data=data.reshape((data.shape[0],data.shape[1] * data.shape[2]))
    # pca
    # pca = PCA(n_components=0.95, svd_solver='full')
    # pca.fit(data)
    # data = pca.fit_transform(data)

    train, test, train_labels, test_labels = train_test_split(data, np.array(data_y),
                                                              stratify=np.array(data_y))
    print("trainsize:", train.shape)
    print("testsize:", test.shape)

    params['classnums'] = max(train_labels) + 1

    train_data = lgb.Dataset(train, label=train_labels)

    param = {'num_leaves': 31, 'objective': 'binary'}
    param['metric'] = 'auc'

    num_round = 10
    bst = lgb.train(param, train_data, num_round)

    bst.save_model(os.path.join(savepath,'model.txt'))
    result = bst.predict(test)

    for i in range(len(result)):
        if result[i]>0.5:
            result[i]=1
        else:
            result[i]=0
    s = metrics.classification_report(test_labels, result, output_dict=True)
    print(metrics.classification_report(test_labels, result))
    df = pd.DataFrame.from_dict(dict(s)).transpose()
    df.to_csv(os.path.join(savepath, 'lgb_result.csv'))

if __name__ == '__main__':
    dataset="data"
    params = {
        "batch_size":256,
        "hidden_dim": 16,
        "mlp_dim":8,  # 前馈的隐藏层
        "depth": 1,  # 前馈的隐藏层
        "attheads": 2,  # num of multihead
        "kernel": 3,
        "cnndepth":1,
        "alpha": 1,
        "temp": 0.7,
        "l2_w":1e-3,
        "l2_reg":0.3,
        "cuda": True

    }

    for _ in range(1):
        now = datetime.datetime.now()
        run_lgb(params, dataset,
            savepath=os.path.join('benchmark', dataset + "_" + str(now) + "_"),
            newcolumns=newcolumns)
