import math

import numpy

import torch
from torch import nn
import loss
import joblib
import causal_cnn
import torch.utils.data as Data
from sklearn.svm import LinearSVC
from sklearn import metrics
from optimiziers import RAdam
from loss import MaskedMSELoss, NoFussCrossEntropyLoss
from tool import EarlyStopping, plot
import matplotlib.pyplot as plt
import  transformer
import pandas as pd
"""
params=
            batch_size,  B
            feature_dim,  C
            time_dim,     L
            hidden_dim,   dim that output from encoder (B,dim)
    
"""


class masknet(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.encoder = masknetencoder(batch_size=params['batch_size'], feature_dim=params['feature_dim'],
                                      time_dim=params['time_dim'],cuda=params['cuda'],
                                      hidden_dim=params['hidden_dim'], mlp_dim=params['mlp_dim'], depth=params['depth'],
                                      attheads=params['attheads'], kernel=params['kernel'])

        self.decoder = masknetdecoder(batch_size=params['batch_size'], feature_dim=params['feature_dim'],
                                      time_dim=params['time_dim'],
                                      hidden_dim=params['hidden_dim'], mlp_dim=params['mlp_dim'], depth=params['depth'],
                                      attheads=params['attheads'], cuda=params['cuda'],kernel=params['kernel'])

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x





class masknet_train(nn.Module):

    def __init__(self,
                 epoch,
                 params,
                 model,
                 lr,
                 cuda,
                 gpu,
                 classifier,
                 earlystopping
                 ):
        super().__init__()
        self.lr=lr
        self.model = model
        self.epoch = epoch
        self.batch_size = params['batch_size']
        self.param = params
        self.cuda = cuda
        self.gpu = gpu
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),lr=lr)
        self.loss = loss  # need!
        self.classifier = classifier
        self.deepcls=deepclassifier(params['hidden_dim'],params['classnums'])
        self.earlystopping=earlystopping

        # init mask matrix (B,C,L) all 0.5
        self.mask_matrix = torch.div(torch.ones(params['batch_size'], params['feature_dim'], params['time_dim']), 2)


    def save_encoder(self, prefix_file):
        """
        Saves the encoder.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        torch.save(
            self.model.get_encoder().state_dict(),
            prefix_file + '_' + self.architecture + '_encoder.pth'
        )
    #def load_encoder(self):
    def save(self, prefix_file):
        """
        Saves the encoder.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_classifier.pkl' and
               '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.save_encoder(prefix_file)
        joblib.dump(
            self.classifier,
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def padding_mask(self,lengths, max_len=None):
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

    def geom_noise_mask_single(self,L, lm, masking_ratio):
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
        keep_mask = numpy.ones(L, dtype=bool)
        p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
        p_u = p_m * masking_ratio / (
                    1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
        p = [p_m, p_u]

        # Start in state 0 with masking_ratio probability
        state = int(numpy.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
        for i in range(L):
            keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
            if numpy.random.rand() < p[state]:
                state = 1 - state

        return keep_mask

    def noise_mask(self,X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
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
                mask = numpy.ones(X.shape, dtype=bool)
                for m in range(X.shape[1]):  # feature dimension
                    if exclude_feats is None or m not in exclude_feats:
                        mask[:, m] = self.geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
            else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
                mask = numpy.tile(numpy.expand_dims(self.geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
        else:  # each position is independent Bernoulli with p = 1 - masking_ratio
            if mode == 'separate':
                mask = numpy.random.choice(numpy.array([True, False]), size=X.shape, replace=True,
                                        p=(1 - masking_ratio, masking_ratio))
            else:
                mask = numpy.tile(numpy.random.choice(numpy.array([True, False]), size=(X.shape[0], 1), replace=True,
                                                p=(1 - masking_ratio, masking_ratio)), X.shape[1])

        return mask

    def acc_count(self,res,label):
        res = res.argmax(dim=-1)
        acount=0
        for i in range(res.shape[0]):
            if res[i]==label[i]:acount+=1
        return acount/res.shape[0]
    def plot_reconstruction(self,x,y):
        """

        :param x: (feature_dim,time_dim) original data
        :param y: (feature_dim,time_dim) reconstructed data
        :return:
        """
        x=x.cpu().detach().numpy()
        y=y.cpu().detach().numpy()
        maxrange=len(x)
        if maxrange<3:
            R=maxrange
        else:
            R=3
        for i in range(R):
            original=x[i]
            reconstrust=y[i]
            plt.figure(figsize=(100,10))
            plt.plot(original, "r", label="original")
            plt.plot(reconstrust, "b", label="reconstrust")
            plt.legend()
            plt.savefig(str(i) + ".jpg")


    def fit_encoder(self, x,test, verbose=False):

        # Check if the given time series have unequal lengths


        train = torch.from_numpy(x)
        test = torch.from_numpy(x)

        train = torch.tensor(train, dtype=torch.float32)
        test = torch.tensor(test, dtype=torch.float32)


        #train_torch_dataset = Data.Dataset(train)
        train_generator = Data.DataLoader(
            train, batch_size=self.batch_size, shuffle=True,num_workers=0
        )
        test_generator=Data.DataLoader(
            test, batch_size=self.batch_size, shuffle=True,num_workers=0
        )
        model = self.model

        criterion = torch.nn.MSELoss(reduction='mean')  # loss还需要设置
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0, last_epoch=-1, verbose=False)

        es=EarlyStopping(self.earlystopping)
        i = 0  # counting epoch
        train_eloss=[]
        test_eloss=[]
        xx=[]
        while i < self.epoch:
            if verbose:
                print('Epoch: ', i + 1)
            xx.append(i+1)
            train_loss=0
            c=0
            for batch in train_generator:
                if self.cuda:
                    model = model.cuda()
                    batch = batch.cuda(self.gpu)
                self.model=self.model.train()


                optimizer.zero_grad()
                batch_result = model(batch)
                loss= criterion(batch_result, batch)
                train_loss = train_loss + loss.item()
                c += 1

                loss.backward()
                #
                optimizer.step()
            print("train_loss:", train_loss/c)
            train_eloss.append(train_loss/c)
            test_loss=0

            c=0
            for test_batch in test_generator:
                if self.cuda:
                    test_batch = test_batch.cuda(self.gpu)

                test_res=model(test_batch)
                test_loss =test_loss+criterion(test_res,test_batch).item()
                c += 1

                #encoder=model.getencoder()
                #test_feature=encoder(test_batch)
                #classifier = LinearSVC()
                #classifier =

            val_loss=test_loss/c
            print("validation_loss: ", val_loss)
            test_eloss.append(val_loss)

            es.__call__(val_loss, model)

            if es.early_stops():
                print("Early stopping")
                # 结束模型训练
                break

            i += 1
        model.load_state_dict(torch.load('checkpoint.pt'))
        self.model=model
        plot(train_loss=train_eloss,test_loss=test_eloss,epoch=xx,name="pretain")




        return model

    def mask_pretain(self, x,test,accumulation_steps, verbose=False):


        train = torch.from_numpy(x)
        test = torch.from_numpy(test)

        train = torch.tensor(train, dtype=torch.float32) # (all,feature_dim,time_dim)
        test = torch.tensor(test, dtype=torch.float32)

        original=train[0]

        train=train.permute(0,2,1) # (all,time_dim,feature_dim)
        test = test.permute(0,2,1)
        masks = self.noise_mask(train[0], masking_ratio=0.15)
        masks=torch.ones(train[0].shape,dtype=torch.bool)


        # masksave=pd.DataFrame(masks)
        # masksave.to_csv("save_mask.csv")

        #masks = torch.from_numpy(masks)






        # train_torch_dataset = Data.Dataset(train)
        train_generator = Data.DataLoader(
            train, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        test_generator = Data.DataLoader(
            test, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        model = self.model

        criterion = MaskedMSELoss()  # loss还需要设置
        optimizer = RAdam(model.parameters(), lr=self.lr
                                     ,weight_decay=1e-8
                                     )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0, last_epoch=-1, verbose=False)

        es = EarlyStopping(self.earlystopping,path="pretain.pt")
        i = 0  # counting epoch
        train_eloss = []
        test_eloss = []
        xx = []
        while i < self.epoch:
            if verbose:
                print('Epoch: ', i + 1)
            xx.append(i + 1)
            train_loss = 0
            c = 0
            model = model.train()
            masks = self.noise_mask(train[0], masking_ratio=0.15)
            masks = torch.from_numpy(masks)
            for batch in train_generator:

                if self.cuda:
                    model = model.cuda()
                    batch = batch.cuda(self.gpu)
                    masks = masks.cuda()



                #batch.masked_fill_(~masks , 0)
                batch_result = model(batch)

                loss = criterion(batch_result, batch,masks)

                train_loss = train_loss + loss.item()
                c += 1

                loss.backward()
                #
                if ((i + 1) % accumulation_steps) == 0:
                    optimizer.step()  # 反向传播，更新网络参数
                    optimizer.zero_grad()
            print("train_loss:", train_loss / c)
            train_eloss.append(train_loss / c)
            test_loss = 0

            c = 0
            model=model.eval()
            for test_batch in test_generator:
                if self.cuda:
                    test_batch = test_batch.cuda(self.gpu)

                test_res = model(test_batch)
                test_loss = test_loss + criterion(test_res, test_batch,masks).item()
                c += 1

                # encoder=model.getencoder()
                # test_feature=encoder(test_batch)
                # classifier = LinearSVC()
                # classifier =

            val_loss = test_loss / c
            print("validation_loss: ", val_loss)
            test_eloss.append(val_loss)

            es.__call__(val_loss, model)

            if es.early_stops():
                print("Early stopping")
                # 结束模型训练
                break

            i += 1
        model.load_state_dict(torch.load('pretain.pt'))
        self.model = model
        plot(train_loss=train_eloss, test_loss=test_eloss, epoch=xx, name="pretain")

        original=original.cuda()
        reconstrustion = model(original.unsqueeze(0).permute(0,2,1)).permute(0,2,1)
        self.plot_reconstruction(original, reconstrustion.squeeze(0))
        plot(train_loss=train_eloss, test_loss=test_eloss, epoch=xx, name="pretain")

        return model

    def fit_maskcls(self,model,train,trainlabel,test,test_label, epoch,biglr,smalllr,verbose=False):
        train = torch.tensor(train, dtype=torch.float32)  # (all,feature_dim,time_dim)
        test = torch.tensor(test, dtype=torch.float32)
        trainlabel = torch.tensor(trainlabel, dtype=torch.long)
        test_label = torch.tensor(test_label, dtype=torch.long)



        train = train.permute(0, 2, 1)  # (all,time_dim,feature_dim)
        test = test.permute(0, 2, 1)

        traindata = Data.TensorDataset(train, trainlabel)
        testdata = Data.TensorDataset(test, test_label)



        train_generator = Data.DataLoader(
            traindata, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        test_generator = Data.DataLoader(
            testdata, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        model = maskcls(model=model,time_dim=self.param["time_dim"],hidden_dim=self.param["hidden_dim"],classnum=self.param["classnums"])

        criterion = NoFussCrossEntropyLoss()
        linear_param = model.output_layer.parameters()
        ignore = list(map(id, linear_param))
        model_param = filter(lambda p: id(p) not in ignore, model.parameters())
        optimizer = RAdam([{'params': model_param}, {'params': linear_param, 'lr': smalllr}], lr=biglr)
        i = 0  # counting epoch
        es = EarlyStopping(self.earlystopping,path="deepcls.pt")

        train_eloss = []
        test_eloss = []
        train_eacc = []
        test_eacc = []
        best_test_acc=0
        xx = []
        if self.cuda:
            model = model.cuda()
        while i < epoch:
            if verbose:
                print('Epoch: ', i + 1)
            xx.append(i + 1)
            train_loss = 0
            train_acc = 0
            c = 0
            model = model.train()
            for batch, (data, target) in enumerate(train_generator):

                if self.cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)

                optimizer.zero_grad()

                batch_result = model(data)
                # target=target.unsqueeze(1)

                loss = criterion(batch_result, target)
                loss=torch.sum(loss)

                train_loss = train_loss + loss.item()
                train_acc += self.acc_count(batch_result, target)

                c += 1

                loss.backward()
                optimizer.step()
            print("train_loss:", train_loss / c)
            train_eloss.append(train_loss / c)
            train_eacc.append(train_acc / c)

            test_loss = 0
            test_acc = 0
            c = 0
            model = model.eval()
            for batch, (data, target) in enumerate(test_generator):

                if self.cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)

                optimizer.zero_grad()

                batch_result = model(data)
                # target=target.unsqueeze(1)

                loss = criterion(batch_result, target)
                loss = torch.sum(loss)

                test_loss = test_loss + loss.item()
                test_acc += self.acc_count(batch_result, target)

                c += 1

            print("validation_loss: ", test_loss / c)
            print("validation_acc: ", test_acc / c)
            test_eloss.append(test_loss/ c)
            test_eacc.append(test_acc/ c)
            es.__call__(test_loss/ c, model)
            if test_acc/ c > best_test_acc:
                best_test_acc=test_acc/ c
                torch.save(model.state_dict(), "bestacc_checkpoint.pt")

            if es.early_stops():
                print("Early stopping")
                # 结束模型训练
                break

            i += 1
        print("best_acc:",best_test_acc,"epoch:",i)
        model.load_state_dict(torch.load('deepcls.pt'))
        plot(train_loss=train_eloss, test_loss=test_eloss, epoch=xx, name="fcls_loss")
        plot(train_loss=train_eacc, test_loss=test_eacc, epoch=xx, name="fcls_acc")
        model=model.cpu()
        result = model(test)
        result = result.argmax(dim=-1)
        result = result.cpu().detach().numpy()
        print(metrics.classification_report(test_label, result))

        return model

    def fit_maskcls_inception(self, model, train, trainlabel, test, test_label, epoch, biglr, smalllr, verbose=False):
        train = torch.tensor(train, dtype=torch.float32)  # (all,feature_dim,time_dim)
        test = torch.tensor(test, dtype=torch.float32)
        trainlabel = torch.tensor(trainlabel, dtype=torch.long)
        test_label = torch.tensor(test_label, dtype=torch.long)

        train = train.permute(0, 2, 1)  # (all,time_dim,feature_dim)
        test = test.permute(0, 2, 1)

        traindata = Data.TensorDataset(train, trainlabel)
        testdata = Data.TensorDataset(test, test_label)

        train_generator = Data.DataLoader(
            traindata, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        test_generator = Data.DataLoader(
            testdata, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        model = maskcls(model=model, time_dim=self.param["time_dim"], hidden_dim=100,
                        classnum=self.param["classnums"])

        criterion = NoFussCrossEntropyLoss()
        linear_param = model.output_layer.parameters()
        ignore = list(map(id, linear_param))
        model_param = filter(lambda p: id(p) not in ignore, model.parameters())
        optimizer = RAdam([{'params': model_param}, {'params': linear_param, 'lr': smalllr}], lr=biglr)
        i = 0  # counting epoch
        es = EarlyStopping(self.earlystopping, path="deepcls.pt")

        train_eloss = []
        test_eloss = []
        train_eacc = []
        test_eacc = []
        best_test_acc = 0
        xx = []
        if self.cuda:
            model = model.cuda()
        while i < epoch:
            if verbose:
                print('Epoch: ', i + 1)
            xx.append(i + 1)
            train_loss = 0
            train_acc = 0
            c = 0
            model = model.train()
            for batch, (data, target) in enumerate(train_generator):

                if self.cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)

                optimizer.zero_grad()

                batch_result = model(data)
                # target=target.unsqueeze(1)

                loss = criterion(batch_result, target)
                loss = torch.sum(loss)

                train_loss = train_loss + loss.item()
                train_acc += self.acc_count(batch_result, target)

                c += 1

                loss.backward()
                optimizer.step()
            print("train_loss:", train_loss / c)
            train_eloss.append(train_loss / c)
            train_eacc.append(train_acc / c)

            test_loss = 0
            test_acc = 0
            c = 0
            model = model.eval()
            for batch, (data, target) in enumerate(test_generator):

                if self.cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)

                optimizer.zero_grad()

                batch_result = model(data)
                # target=target.unsqueeze(1)

                loss = criterion(batch_result, target)
                loss = torch.sum(loss)

                test_loss = test_loss + loss.item()
                test_acc += self.acc_count(batch_result, target)

                c += 1

            print("validation_loss: ", test_loss / c)
            print("validation_acc: ", test_acc / c)
            test_eloss.append(test_loss / c)
            test_eacc.append(test_acc / c)
            es.__call__(test_loss / c, model)
            if test_acc / c > best_test_acc:
                best_test_acc = test_acc / c
                torch.save(model.state_dict(), "bestacc_checkpoint.pt")

            if es.early_stops():
                print("Early stopping")
                # 结束模型训练
                break

            i += 1
        print("best_acc:", best_test_acc, "epoch:", i)
        model.load_state_dict(torch.load('deepcls.pt'))
        plot(train_loss=train_eloss, test_loss=test_eloss, epoch=xx, name="fcls_loss")
        plot(train_loss=train_eacc, test_loss=test_eacc, epoch=xx, name="fcls_acc")
        model = model.cpu()
        result = model(test)
        result = result.argmax(dim=-1)
        result = result.cpu().detach().numpy()
        print(metrics.classification_report(test_label, result))

        return model

    def evaluate(self, test,test_label, model):

        test = torch.tensor(test, dtype=torch.float32)
        test_label = torch.tensor(test_label, dtype=torch.float32)
        model=model.eval()
        model = model.cpu()
        result = model(test)
        result = result.argmax(dim=-1)
        result = result.cpu().detach().numpy()
        print(metrics.classification_report(test_label, result))





    def fit_deepclassifier(self,train,trainlabel,test,test_label, epoch,lr,verbose=False,cearlystopping=15):
        train = torch.tensor(train, dtype=torch.float32)
        test = torch.tensor(test, dtype=torch.float32)

        trainlabel=torch.tensor(trainlabel,dtype=torch.long)
        test_label=torch.tensor(test_label,dtype=torch.long)

        train =  Data.TensorDataset(train,trainlabel)


        # train_torch_dataset = Data.Dataset(train)
        train_generator = Data.DataLoader(
            train, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        model = self.deepcls




        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        i = 0  # counting epoch
        es = EarlyStopping(cearlystopping)

        train_eloss = []
        train_eacc = []
        test_eloss = []
        test_eacc=[]
        xx = []

        while i < epoch:
            if verbose:
                print('Epoch: ', i + 1)
            xx.append(i+1)
            train_loss = 0
            train_acc=0
            c=0
            for batch,(data, target) in enumerate(train_generator):
                if self.cuda:
                    model = model.cuda()
                self.model = self.model.train()
                if self.cuda:
                    data = data.cuda(self.gpu)
                    target=target.cuda(self.gpu)

                optimizer.zero_grad()

                batch_result = model(data)
                #target=target.unsqueeze(1)

                loss = criterion(batch_result, target)

                train_loss = train_loss + loss.item()
                train_acc += self.acc_count(batch_result, target)
                c+=1


                loss.backward()
                optimizer.step()
            print("train_loss:", train_loss / c)
            train_eloss.append(train_loss / c)
            train_eacc.append(train_acc / c)
            model=model.cpu()
            test_res = model(test)
            test_res= test_res
            test_loss = criterion(test_res, test_label).item()
            test_acc = self.acc_count(test_res, test_label)

            print("validation_loss: ", test_loss)
            print("validation_acc: ", test_acc)
            test_eloss.append(test_loss)
            test_eacc.append(test_acc)
            es.__call__(test_loss, model)

            if es.early_stops():
                print("Early stopping")
                # 结束模型训练
                break

            i += 1
        model.load_state_dict(torch.load('checkpoint.pt'))
        plot(train_loss=train_eloss, test_loss=test_eloss, epoch=xx, name="clsloss")
        plot(train_loss=train_eacc,test_loss=test_eacc,epoch=xx,name="acc")

        return model


    def fit_fdeepmodel(self,train,trainlabel,test,test_label,pretain, epoch,biglr,smalllr,verbose=False):
        train = torch.tensor(train, dtype=torch.float32)
        test = torch.tensor(test, dtype=torch.float32)

        etest_label = test_label

        trainlabel=torch.tensor(trainlabel,dtype=torch.long)
        test_label=torch.tensor(test_label,dtype=torch.long)

        if self.cuda:
            test = test.cuda(self.gpu)

        etest = test
        train =  Data.TensorDataset(train,trainlabel)


        # train_torch_dataset = Data.Dataset(train)
        train_generator = Data.DataLoader(
            train, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        #test_generator = Data.DataLoader(
        #    test, batch_size=self.batch_size, shuffle=True, num_workers=0
        #)
        model = deepfmodel(pretain_model=pretain,input_dim=self.param['hidden_dim'],classnum=self.param['classnums'])
        if self.cuda:
            model = model.cuda()


        criterion = torch.nn.CrossEntropyLoss()
        linear_param=model.cls.parameters()
        ignore=list(map(id,linear_param))
        model_param=filter(lambda p:id(p) not in ignore,model.parameters())
        optimizer = torch.optim.Adam([{'params': model_param},{'params': linear_param,'lr':smalllr}], lr=biglr)

        i = 0  # counting epoch
        es = EarlyStopping(self.earlystopping)

        train_eloss = []
        test_eloss = []
        train_eacc=[]
        test_eacc=[]
        xx = []

        while i < epoch:
            if verbose:
                print('Epoch: ', i + 1)
            xx.append(i+1)
            train_loss = 0
            train_acc = 0
            c=0
            for batch,(data, target) in enumerate(train_generator):
                self.model = self.model.train()
                if self.cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)

                optimizer.zero_grad()

                batch_result = model(data)
                #target=target.unsqueeze(1)

                loss = criterion(batch_result, target)

                train_loss = train_loss + loss.item()
                train_acc +=self.acc_count(batch_result,target)

                c+=1

                loss.backward()
                optimizer.step()
            print("train_loss:", train_loss / c)
            train_eloss.append(train_loss / c)
            train_eacc.append(train_acc / c)



            test_res = model(test )
            test_res = test_res.cpu()
            test_loss = criterion(test_res, test_label).item()
            test_acc = self.acc_count(test_res,test_label)




            print("validation_loss: ", test_loss)
            print("validation_acc: ",test_acc)
            test_eloss.append(test_loss)
            test_eacc.append(test_acc)
            es.__call__(test_loss, model)

            if es.early_stops():
                print("Early stopping")
                # 结束模型训练
                break

            i += 1
        model.load_state_dict(torch.load('checkpoint.pt'))
        plot(train_loss=train_eloss, test_loss=test_eloss, epoch=xx, name="fcls_loss")
        plot(train_loss=train_eacc,test_loss=test_eacc,epoch=xx,name="fcls_acc")

        result = model(etest)
        result = result.argmax(dim=-1)
        result = result.cpu().detach().numpy()
        print(metrics.classification_report(etest_label, result))

        return model






    def Encoder(self, x):
        batch_size = self.batch_size
        if not torch.is_tensor(x):
               x=torch.from_numpy(x)
               x=torch.tensor(x, dtype=torch.float32)

        #feature = Data.DataLoader(x)
        feature_generator = Data.DataLoader(
            x, batch_size=batch_size
        )
        features = numpy.zeros((numpy.shape(x)[0], self.param['hidden_dim']))
        #features=numpy.zeros(numpy.shape(x))
        encoder = self.model.get_encoder()
        encoder = encoder.eval()

        count = 0
        if self.cuda:
            encoder=encoder.cuda()
        with torch.no_grad():
            for batch in feature_generator:
                if self.cuda:

                    batch = batch.cuda(self.gpu)
                features[
                count * batch_size: (count + 1) * batch_size
                ] = encoder(batch).cpu()
                count += 1

        return features

    #def save_feature(self,feature):
    def fit_classifier(self, train, train_label, test, test_label):
        classifier = LinearSVC()

        classifier.fit(train, train_label)

        result = classifier.predict(test)
        acc = metrics.accuracy_score(test_label, result)

        print("test accuracy : ", acc)
        print(metrics.classification_report(test_label, result))

    #def build_finetune_cls(self,model):




    def save_mask(self, prefix_file):
        """
        save mask matrix as important output
        :param prefix_file:
        :return:
        """
        torch.save(self.mask_matrix, prefix_file, "mask.pt")

class maskcls(nn.Module):
    """
    model ouput= (batch_size, seq_length, d_model)

    """

    def __init__(self, time_dim, hidden_dim, classnum,model):
        super(maskcls, self).__init__()

        self.output_layer=nn.Sequential(
            nn.Linear(hidden_dim * time_dim, time_dim),
            nn.ReLU(),
            nn.BatchNorm1d(time_dim, eps=1e-5),
            nn.Dropout(0.3),
            nn.Linear(time_dim, classnum)
        )


        self.model=model
        self.softmax=nn.Softmax()

    def forward(self,x):
        """
        :param x: (batch_size, seq_length, d_model)
        :return: (batch_size, classnum)
        """
        x=self.model(x)


        output = x.reshape(x.shape[0], -1)

        output = self.output_layer(output)
        #output = self.softmax(output)
        return output






class linear_add_layer(nn.Module):

    def __init__(self, batch_size, hidden_dim):
        """

        :param batch_size:
        :param hidden_dim:
        """
        super(linear_add_layer, self).__init__()
        self.g=torch.nn.GELU()
        #self.bn=nn.BatchNorm1d(num_features=hidden_dim)
        #self.params = nn.ParameterList([nn.Parameter(torch.randn(batch_size, hidden_dim))for i in  range(2)])

    def forward(self, global_f, local_f):

        x = global_f + local_f
        #x = self.g(x)
        return x


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, net):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.net = net

    def forward(self, x, **kwargs):
        #    x = torch.tensor(x, dtype=torch.float32)
        return self.net(self.norm(x), **kwargs)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.num_heads = num_heads
        self.scale = dim_per_head ** -0.5

        inner_dim = dim_per_head * num_heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.attend = nn.Softmax(dim=-1)

        project_out = not (num_heads == 1 and dim_per_head == dim)
        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, l, d = x.shape

        '''i. QKV projection'''
        # (b,l,dim_all_heads x 3)

        qkv = self.to_qkv(x)

        # (3,b,num_heads,l,dim_per_head)
        qkv = qkv.view(b, l, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # 3 x (1,b,num_heads,l,dim_per_head)
        q, k, v = qkv.chunk(3)
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)

        '''ii. Attention computation'''
        attn = self.attend(
            torch.matmul(q, k.transpose(-1, -2)) * self.scale
        )

        '''iii. Put attention on Value & reshape'''
        # (b,num_heads,l,dim_per_head)
        z = torch.matmul(attn, v)
        # (b,num_heads,l,dim_per_head)->(b,l,num_heads,dim_per_head)->(b,l,dim_all_heads)
        z = z.transpose(1, 2).reshape(b, l, -1)
        # assert z.size(-1) == q.size(-1) * self.num_heads

        '''iv. Project out'''
        # (b,l,dim_all_heads)->(b,l,dim)
        out = self.out(z)
        # assert out.size(-1) == d

        return out


class Transformer(nn.Module):
    def __init__(self, dim, mlp_dim, feature_dim,depth=6, num_heads=8,  dropout=0.3,cuda=False):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, num_heads=num_heads, dropout=dropout)),
                #SelfAttention(dim, num_heads=num_heads, dropout=dropout),
                PreNorm(dim, FFN(dim, mlp_dim, dropout=dropout))
                #FFN(dim, mlp_dim, dropout=dropout),
                #nn.BatchNorm1d(feature_dim),
            ]))

    def forward(self, x):
        #print(x.shape)
        for norm_attn,norm_ffn in self.layers:
            x = x + norm_attn(x)

            x = x + norm_ffn(x)
            #x = x + b2(x)

        return x


class positional_encoding(nn.Module):
    """
    shape=(B,C,L）
    """

    def __init__(self, shape, dropout=0.1,cuda=False):
        super().__init__()
        C = shape[1]
        L = shape[2]
        #self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(C, L)

        position = torch.arange(0, C, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, L, 2).float() * (-math.log(10000.0) / L))
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_ len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)              # [1,max_len, d_model]
        if cuda:
            pe=pe.cuda()
        print(pe.device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #print(x.size())
        #print(self.pe.size())
        pe = self.pe


        pe=pe.to(x.device)
        x = x + self.pe      #[:x.size(), :]
        return x


class masknetencoder(nn.Module):
    def __init__(
            self,
            batch_size,
            feature_dim,
            time_dim,
            hidden_dim,
            mlp_dim,
            depth,
            attheads,
            kernel,
            cuda=False

    ):
        super().__init__()
        self.cnn = causal_cnn.CausalCNNEncoder(
            in_channels=feature_dim,
            channels=feature_dim,
            depth=3,
            reduced_size=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel
        )

        self.globaltransformer = Transformer(time_dim, mlp_dim,feature_dim=feature_dim, depth=depth, num_heads=attheads)
        #self.globaltransformer=transformer.TransformerBatchNormEncoderLayer

        self.addlayer = linear_add_layer(batch_size=batch_size,hidden_dim=hidden_dim )
        self.encoder_global = nn.Sequential(
            positional_encoding(shape=[batch_size, feature_dim, time_dim],cuda=cuda),
            self.globaltransformer,
            torch.nn.GELU(),
        )  # (batch_size,feature_dim,time_dim)->(batch_size,hidden_dim)
        #self.linearproj=nn.Linear(hidden_dim*2,hidden_dim,bias=False)
        #self.relu=nn.ReLU()


    def forward(self, x):
        global_feature = self.encoder_global(x)
        global_feature=self.cnn(global_feature)

        local_feature = self.cnn(x)
        all_feature = self.addlayer(global_feature, local_feature)    #再想想特征组合的方式

        return all_feature  # (batch_size,hidden_dim)


class masknetdecoder(nn.Module):
    def __init__(self,
                 batch_size,
                 feature_dim,
                 time_dim,
                 hidden_dim,
                 mlp_dim,
                 depth,
                 attheads,
                 kernel,
                 cuda=False,
                 dropout=0.1
                 ):
        super().__init__()
        self.pos=positional_encoding(shape=[batch_size, feature_dim, time_dim],cuda=cuda)
        self.transformer_decoder = Transformer(time_dim, mlp_dim,feature_dim=feature_dim, depth=depth, num_heads=attheads)

        updim=time_dim/2

        updim=int(updim)
        if updim%2!=0:
            updim=updim+1
        #self.upsampling_linearprob = nn.Linear(1,updim,
        #                                      bias=False)  # (batch_size,hidden_dim,1)->(batch_size,hidden_dim,time_dim/2)
        #self.upsampling = causal_cnn.CausalConvTransblock(in_channels=hidden_dim, out_channels=feature_dim,Lin= updim, time_dim=time_dim, kernel_size=kernel,
        #                                                 dilation=1)  # (batch_size,hidden_dim,feature_dim)->(batch_size,feature_dim,time_dim)
        self.upsampling=nn.Sequential(
            nn.Linear(1, updim),  # (batch_size,hidden_dim,1)->(batch_size,hidden_dim,time_dim/2)
            #nn.BatchNorm2d(num_features=updim),
            #nn.Dropout(dropout),
            causal_cnn.CausalConvTransblock(in_channels=hidden_dim, out_channels=feature_dim, Lin=updim,time_dim=time_dim, kernel_size=kernel, dilation=1),
            #nn.BatchNorm2d(num_features=feature_dim)
        )


    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.upsampling(x)

        x = self.pos(x)
        x = self.transformer_decoder(x)
        return x

class deepclassifier(nn.Module):
    def __init__(self,input_dim,classnum,dropout=0.1):
        super().__init__()

        self.classify=nn.Softmax(dim=-1)




        self.out=nn.Sequential(
            #nn.Conv1d(in_channels=input_dim, out_channels=1,kernel_size=3,padding='SAME'),
            nn.Linear(input_dim,classnum+1)
        )





    def forward(self,x):
        x=self.out(x)

        #x=x.squeeze(-1)
        class_res=self.classify(x)
        #class_res=class_res.squeeze(-1)


        return class_res

class deepfmodel(nn.Module):
    def __init__(self,pretain_model,input_dim,classnum,dropout=0.1):
        super().__init__()
        self.big=pretain_model
        self.cls=deepclassifier(input_dim=input_dim,classnum=classnum,dropout=dropout)

    def forward(self,x):
        x=self.big(x)
        x=self.cls(x)
        return x



