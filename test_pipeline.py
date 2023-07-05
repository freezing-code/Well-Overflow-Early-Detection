import datetime
import os.path
import utils
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch

from pipeline import  masknet_train
from mymodels import MTSTco,TCNST
import tool

import json

newcolumns = ['扭矩(KN.m)', '泵冲2(spm)', '立压log(MPa)', '大钩负荷(KN)', '大钩位置(m)',
              '平均钻速(m/min)', '平均钻压(KN)', '入口流量log(L/s)', '出口流量log(%)', '总池体积(m3)', '迟到时间(min)', '定点压耗(MPa)',
              '迟到时间(min)', '入口温度(℃)', '出口温度(℃)', '总烃(%)', 'PWD垂深(m)', 'PWD环空压力(MPa)', 'PWD井斜(deg)', 'PWD方位(deg)',
              'C2(%)', '目标回压(MPa)']

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def run(params, dataset, savepath, augmented_path,newcolumns,data_augmented="new",augmented_size=2):
    data_x,data_y,data_next_x=utils.readdata(dataset,newcolumns)
    train = np.array(data_x)
    next_train = np.array(data_next_x)

    params['feature_dim'] = train.shape[2]
    params['time_dim'] = train.shape[1]
    params['classnums'] = max(data_y) + 1
    train = train.transpose(0, 2, 1)
    next_train = next_train.transpose(0, 2, 1)

    model = MTSTco(feat_dim=params["feature_dim"], max_len=params["time_dim"], d_model=params["hidden_dim"],
                   n_heads=params["attheads"],
                   num_layers=params["depth"], dim_feedforward=params["mlp_dim"], pos_encoding="learnable",
                   cnnhidden_channels=params["hidden_dim"], cnndepth=params["cnndepth"], kernel_size=params["kernel"])

    # model = TCNST(feat_dim=params["feature_dim"], max_len=params["time_dim"], d_model=params["hidden_dim"],
    #                 n_heads=params["attheads"],
    #                 num_layers=params["depth"], dim_feedforward=params["mlp_dim"], pos_encoding="learnable",
    #                 cnnhidden_channels=params["hidden_dim"], cnndepth=params["cnndepth"], kernel_size=params["kernel"])
    factory = masknet_train(epoch=200, params=params, model=model, lr=0.01, cuda=True, gpu=0,
                            classifier=None, earlystopping=40)
    pretain = factory.mask_pretain(train, next_train, accumulation_steps=1, verbose=True, savepath=savepath)

    train, test, train_labels, test_labels = train_test_split(np.array(data_x), np.array(data_y),
                                                              stratify=np.array(data_y))
    print("trainsize:", train.shape)
    print("testsize:", test.shape)


    params['classnums'] = max(train_labels) + 1
    train = train.transpose(0, 2, 1)
    test = test.transpose(0, 2, 1)
    torch.cuda.empty_cache()
    # summary(pretain,(train.shape[2],train.shape[1]))
    augmented_save_path = os.path.join(augmented_path + dataset)
    if data_augmented == "new":
        newdata=None
        newdata_label=None
        tool.mkdir(augmented_save_path)
        neg_index=np.where(train_labels==1)
        newdata,newdata_label=factory.data_augment(train[neg_index],train_labels[neg_index],augmented_size,augmented_save_path,masking_ratio=0.1)


        # summary(pretain,(train.shape[2],train.shape[1]))
        train = np.concatenate((train,newdata))
        train_labels = np.concatenate((train_labels,newdata_label))
    elif data_augmented == "old":
        if augmented_path == None:
            print("augmented data path is None")
            return
        else:
            print("loading from",augmented_save_path)
            newdata=torch.load(os.path.join(augmented_save_path,"augmented_Data"))
            newdata_label=torch.load(os.path.join(augmented_save_path,"augmented_Label"))
            train = np.concatenate((train, newdata.detach().numpy()))
            train_labels = np.concatenate((train_labels, newdata_label.detach().numpy()))
    else:
        print("no augmented")

    positive_ratio = np.sum(train_labels) / train_labels.shape[0]
    negative_ratio = 1 - positive_ratio
    print("weight:", 1 / positive_ratio, 1 / negative_ratio)
    print("train normal/abnormal : ",np.sum(train_labels)," / ",train_labels.shape[0]-np.sum(train_labels))
    print("test normal/abnormal : ", np.sum(test_labels), " / ", test_labels.shape[0] - np.sum(test_labels))
    sample_weighted_list = [    ]
    for label in train_labels:
        if label == 1:
            sample_weighted_list.append(1 / positive_ratio)
        else:
            sample_weighted_list.append(1 / negative_ratio)

    deepf, acc = factory.fit_maskcls_coce(pretain.TSTencoder, train, train_labels, test, test_labels, epoch=500,
                                          biglr=0.01, smalllr=0.01, verbose=True,early_stop=50, savepath=savepath,samples_weight=sample_weighted_list)
    df = pd.DataFrame.from_dict(params, orient='index').transpose()
    df.to_csv(os.path.join(savepath, 'params.csv'))

    os.rename(savepath, savepath + str(acc))


if __name__ == '__main__':
    dataset="data"
    params = {
        "batch_size":256,
        "hidden_dim": 16,
        "mlp_dim":8,  # 前馈的隐藏层
        "depth": 1,  # 前馈的隐藏层
        "attheads": 2,  # num of multihead
        "kernel": 3,
        "cnndepth":3,
        "alpha": 1,
        "temp": 0.7,
        "l2_w":1e-3,
        "l2_reg":0.3,
        "cuda": True

    }

    for _ in range(5):
        now = datetime.datetime.now()
        run(params, dataset,
            savepath=os.path.join('benchmark', dataset + "_" + str(now) + "_"),
            augmented_path="augmented_data", data_augmented="no",newcolumns=newcolumns)
