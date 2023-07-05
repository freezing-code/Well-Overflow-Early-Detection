import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(c_in, c_out, ks=10, sd=1):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=ks),  # nn.Conv1d(in_channels=c_in,out_channels=c_out,kernel_size=ks)
        nn.BatchNorm1d(c_out, momentum=0.5),
        nn.ReLU(True),
        nn.MaxPool1d(kernel_size=2)
    )

class CNNLSTM(nn.Module):
    def __init__(self, xcol_size):
        super(CNNLSTM, self).__init__()
        self.xcol_size = xcol_size
        # x shape(500,56)
        self.conv1 = conv(xcol_size, 64, 11)  # (245,64)
        # x shape(245,64)
        self.conv2 = conv(64, 128, 7)  # (119,128)
        # x shape(119,128)
        self.conv3 = conv(128, 128, 10)  # （55,128)
        # x shape(55,128) 256, 117, 128
        self.lstm = nn.LSTM(
            input_size=128,  # 每行的数据个数
            hidden_size=50,  # hidden unit
            num_layers=1,
            batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )  # output shape (55,50)
        self.dense1 = nn.Linear(5850, 500)
        self.dropout = nn.Dropout(0.3)  # drop 50% of the neuron
        self.dense2 = nn.Linear(500, 200)
        self.dense3 = nn.Linear(200, 2)
        self.softmax = nn.Softmax(dim=1)

        self.output_layer=nn.Sequential(
            nn.Linear(5850, 500),
            nn.Dropout(0.3),
            nn.Linear(500, 200),
            nn.Dropout(0.3),
            nn.Linear(200, 2)
        )

    def forward(self, x):
        # input: (batch,time,feature)
        # torch.Size([100, 500, 56])
        # print("cnnlstm input size: " )

        x = x.permute(0, 2, 1)  # (batch,feature,time)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.permute(0, 2, 1) #(batch,time,feature

        x, _ = self.lstm(x, None)
        x = x.flatten(start_dim=1)# 展平多维

        x =self.output_layer(x)

        return x

    # 二分类
    def predict(self, x):
        output1 = self.forward(x)
        output = self.softmax(output1)
        pred = output.max(1, keepdim=True)[1].squeeze(1)  # get the index of the max log-probability
        return pred
