import random

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA

import os
import math

from torch.utils.data import TensorDataset, DataLoader


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def early_stops(self):
        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class plot():
    def __init__(self, train_loss, test_loss, epoch, name,val_loss=None):
        plt.figure()
        plt.plot(epoch, train_loss, "r", label="train")
        if test_loss is not None:
            plt.plot(epoch, test_loss, "b", label="test")
        if val_loss is not None:
            plt.plot(epoch, val_loss, "b", label="valid")
        plt.plot()
        plt.legend()
        plt.savefig(name + ".jpg")


class tsne():
    def __init__(self, n, perpxity, iteration, init='pca'):
        self.tsne = manifold.TSNE(n_components=n, perplexity=perpxity, n_iter=iteration, init=init)

    def show2d(self, x, y):
        X_tsne = self.tsne.fit_transform(x)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()

    # def show3d(self,x,y):

    def plot_embedding_3d(self, X, y, title=None):  # 坐标缩放到[0,1]区间

        x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
        X = (X - x_min) / (x_max - x_min)
        # 降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        for i in range(X.shape[0]):
            ax.text(X[i, 0], X[i, 1], X[i, 2], str(y[i]), color=plt.cm.Set1(y[i] / 10.),
                    fontdict={'weight': 'bold', 'size': 9})

        if title is not None: plt.title(title)

        plt.show()

    def pca(self, x, y):
        p = PCA(n_components=2)
        X_tsne = p.fit_transform(x)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()


def load_UEA_DATA(path, dataset):
    train = torch.load(path + '/' + dataset + '/' + 'train')
    train_labels = torch.load(path + '/' + dataset + '/' + 'train_labels')
    test = torch.load(path + '/' + dataset + '/' + 'test')
    test_labels = torch.load(path + '/' + dataset + '/' + 'test_labels')

    return train.numpy(), train_labels.numpy(), test.numpy(), test_labels.numpy()


class HookTool:
    def __init__(self):
        self.fea = None
        self.feain = None

    def hook_fun(self, module, fea_in, fea_out):
        '''
        注意用于处理feature的hook函数必须包含三个参数[module, fea_in, fea_out]，参数的名字可以自己起，但其意义是
        固定的，第一个参数表示torch里的一个子module，比如Linear,Conv2d等，第二个参数是该module的输入，其类型是
        tuple；第三个参数是该module的输出，其类型是tensor。注意输入和输出的类型是不一样的，切记。
        '''
        self.fea = fea_out


def acc_count(res, label):
    res = res.argmax(dim=-1)
    acount = 0
    for i in range(res.shape[0]):
        if res[i] == label[i]: acount += 1
    return acount / res.shape[0]


def plot_reconstruction(x, y, savepath):
    """

        :param x: (feature_dim,time_dim) original data
        :param y: (feature_dim,time_dim) reconstructed data
        :return:
        """
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    dim_limit=x.shape[0]
    wants=min(3,dim_limit)
    for i in range(wants):
        original = x[i]
        reconstrust = y[i]
        plt.figure()
        plt.plot(original, "r", label="original")
        plt.plot(reconstrust, "b", label="reconstrust")
        plt.legend()
        plt.savefig(os.path.join(savepath, str(i) + ".jpg"))


def adddic(dic, trainindex, train_label, dic_size):
    """

    :param dic: old dic
    :param trainindex: index of data
    :param train_label:  label of data
    :param dic_size : sample for each label or add how many sample to dic
    :return: dic:['data_index','labels'],trainindex
    """
    classnum = max(train_label) + 1

    # creating select pools
    pools = [[] for _ in range(classnum)]

    row = train_label.shape[0]
    for i in range(row):
        pools[train_label[i]].append(i)

    if not dic:
        dic = pd.DataFrame(columns=['data_index', 'labels'])

    for label in range(classnum):

        # sampling from data
        pool = pools[label]
        n = len(pool)
        selected = random.sample(pool, dic_size)

        for select in selected:
            trainindex.remove(select)

        newdataframe = pd.DataFrame(columns=['data_index', 'labels'])
        newdataframe['data_index'] = selected
        newdataframe['labels'] = [label for _ in range(dic_size)]
        # add to dic
        dic = pd.concat([dic, newdataframe], axis=0)

    return dic, trainindex


def traindataprepare(train, train_labels, dic_size):
    """

    :param train:
    :param train_labels:
    :return: dic:['data_index','labels'],dicdata,newtrain:selected train data
    """
    samplesize = train.shape[0]
    sampleindex = [i for i in range(samplesize)]
    # 每一个index代表着数据的某一个位置

    dicindex, trainindex = adddic(None, sampleindex, train_labels, dic_size)

    # 根据index重新选择train 和 dic

    dicdata = train[dicindex['data_index'].tolist()]
    newtrain = train[trainindex]
    newtrain_labels = train_labels[trainindex]
    print("dictionary size: ", dicdata.shape)
    print("selected train size: ", train.shape)

    return dicindex, dicdata, newtrain, torch.tensor(newtrain_labels)


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (
            1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


def noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask


def cuttinglist(l, size):
    lower = 0
    up = lower + size
    listoflists = []
    gate = len(l)
    print("current list size is ", gate, "and your size is ", size)

    if gate % size:
        print("it will appear a small set in size of ", gate % size)

    while up <= gate:
        listoflists.append(l[lower:up])
        lower = up
        up = lower + size
    # back to last step
    if up == gate + size:
        return listoflists
    else:
        # 填充一些重复的部分
        lower = up - size
        up = gate
        smallset = l[lower:up]
        need = lower + size - gate
        extra = random.sample(range(0, gate), need)
        for e in extra:
            smallset.append(l[e])
        listoflists.append(smallset)
        print("fetching ", len(listoflists), " lists in total")
    return listoflists


def symlist(listoflists):
    """
    :param listoflists: [class_num,class_split_list]
          there will be some set has a different size, we look them up to the max size
    :return: symlists :the size should be the same in all class
    """

    row = len(listoflists)
    # checking the max
    max_list_size = 0
    for i in range(row):
        temp = len(listoflists[i])
        if temp > max_list_size:
            max_list_size = temp

    for i in range(0, row):
        current_size = len(listoflists[i])
        if current_size < max_list_size:
            need_size = max_list_size - current_size
            current=listoflists[i]
            for _ in range(need_size):
                current.append(current[random.randint(0,current_size-1)])
            listoflists[i]=current

    return listoflists


def batchcreator(train_data, label, sampleforeachclass=3, batch_scale=1):
    # 要求每个类至少有sampleforeachclass个样本
    # 检查data中是否有类别是不够sampleforeachclass的
    checklabels = {}
    classnum = max(label) + 1
    print(label)
    for l in label:
        old = checklabels.get(int(l), 0)
        checklabels[int(l)] = old + 1
    print(checklabels)
    key_min = min(checklabels.keys(), key=(lambda k: checklabels[k]))
    min_class_num = checklabels[key_min]
    print("the less class has : ", min_class_num, " samples")
    if sampleforeachclass > min_class_num:
        sampleforeachclass = min_class_num
        print("readjusting class sample number to :", sampleforeachclass)
    batch_size = classnum * sampleforeachclass * batch_scale
    print("batch preparing, batchsize : ", batch_size)

    # 开始抽选batch，每个batch保证有class总数
    rows = train_data.shape[0]

    data_index = torch.tensor([i for i in range(rows)])

    classindexlist = []
    for c in range(classnum):
        indexs = data_index[label == c].tolist()
        print(indexs)

        random.shuffle(indexs)

        classindexlist.append(cuttinglist(indexs, sampleforeachclass))

    # classindexlist 存着每个class等分后的结果
    print("class num is ",len(classindexlist))
    for indexnum in range(len(classindexlist)):
            print("checking border",  len(classindexlist[indexnum]))

    classindexlist=symlist(classindexlist)

    print("checking border after sym", len(classindexlist), len(classindexlist[0]))

    batch_list = []

    batchnum = len(classindexlist[0])

    for j in range(batchnum):
        onebatch = []
        for c in range(0, classnum):
            print(c, j)
            onebatch.extend(classindexlist[c][j])

        batch_list.append(onebatch)

    batch_data = None
    batch_label = None
    for batch in batch_list:
        batchsample = torch.tensor(train_data[batch, :, :], dtype=torch.float32)
        batchlabel = torch.tensor(label[batch], dtype=torch.long)

        if batch_data is not None:
            batch_data = torch.cat((batch_data, batchsample))
            batch_label = torch.cat((batch_label, batchlabel))
        else:
            batch_data = batchsample
            batch_label = batchlabel

    trainset = TensorDataset(batch_data, batch_label)
    train_generator = DataLoader(
        trainset, batch_size=int(batch_size), shuffle=False, num_workers=0
    )

    return train_generator


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("creating saving to", path)

    else:
        print(path, " already existed")
