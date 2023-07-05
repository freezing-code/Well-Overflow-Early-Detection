import math
import os


import numpy
import numpy as np

import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

import loss
import joblib
import causal_cnn
import torch.utils.data as Data
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
import tool
from masknet import maskcls
from optimiziers import RAdam
from loss import MaskedMSELoss, NoFussCrossEntropyLoss, InfoNCE, Contrastiveloss,l2_reg_loss,metricloss
from tool import EarlyStopping, plot, noise_mask, plot_reconstruction, acc_count
import matplotlib.pyplot as plt
import transformer
import pandas as pd

global_norm=10

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
        self.lr = lr
        self.model = model
        self.epoch = epoch
        self.batch_size = params['batch_size']
        self.param = params
        self.cuda = cuda
        self.gpu = gpu
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.loss = loss  # need!
        self.classifier = classifier
        self.earlystopping = earlystopping



        # init mask matrix (B,C,L) all 0.5
        self.mask_matrix = torch.div(torch.ones(params['batch_size'], params['feature_dim'], params['time_dim']), 2)

    def mask_pretain(self, x, next_x, accumulation_steps, savepath, verbose=False,use_unseen=False):

        train = torch.from_numpy(x)
        train_next = torch.from_numpy(next_x)

        train = torch.tensor(train, dtype=torch.float32)  # (all,feature_dim,time_dim)
        train_next = torch.tensor( train_next, dtype=torch.float32)
        original = train[0]

        train = train.permute(0, 2, 1)  # (all,time_dim,feature_dim)
        train_next =  train_next.permute(0, 2, 1)



        # train_torch_dataset = Data.Dataset(train)
        train_pair=Data.TensorDataset(train, train_next)
        train_generator = Data.DataLoader(
            train_pair, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        model = self.model

        criterion = MaskedMSELoss()  # loss还需要设置
        optimizer = RAdam(model.parameters(), lr=self.lr
                          , weight_decay=1e-4
                          )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0, last_epoch=-1, verbose=False)
        savedic = os.path.join(savepath, "pretain_model")
        tool.mkdir(savedic)

        espath = os.path.join(savedic, "pretain.pt")
        es = EarlyStopping(self.earlystopping, path=espath)

        masks = noise_mask(train[0], masking_ratio=0.15)
        # masks = torch.ones(train[0].shape, dtype=torch.bool)

        masksave = pd.DataFrame(masks)
        masksave.to_csv(os.path.join(savedic, "save_mask.csv"))

        masks = torch.from_numpy(masks)

        i = 0  # counting epoch
        train_eloss = []
        test_eloss = []
        epochcounter = []
        mask_log = torch.zeros(train[0].shape, dtype=torch.bool)
        # epoch_mask_log = None
        # 测试 固定mask
        # masks = torch.ones(train[0].shape, dtype=torch.bool)
        while i < self.epoch:
            if verbose:
                print('Epoch: ', i + 1)
            epochcounter.append(i + 1)
            train_loss = 0
            c = 0
            model = model.train()
            # masks = noise_mask(train[0], masking_ratio=0.15)
            # masks = torch.from_numpy(masks)
            # int_masks = np.array(masks, dtype=np.int)
            # mask_log = mask_log + int_masks
            # no masking


            for batch, (data, target) in enumerate(train_generator):

                if self.cuda:
                    model = model.cuda()
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)
                    masks = masks.cuda()

                # masking data
                data.masked_fill_(~masks, 0)

                batch_result = model(data)
                # print(embedding.shape,batch_result.shape)

                loss = criterion(batch_result, target, masks)



                train_loss = train_loss + loss.item()
                c += 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=global_norm)
                #
                if ((i + 1) % accumulation_steps) == 0:
                    optimizer.step()  # 反向传播，更新网络参数
                    optimizer.zero_grad()
            print("train_loss:", train_loss / c)
            train_eloss.append(train_loss / c)
            test_loss = 0

            c = 0
            model = model.eval()

            es.__call__(train_loss, model)

            if es.early_stops():
                print("Early stopping")
                # 结束模型训练
                break

            i += 1
        model.load_state_dict(torch.load(espath))
        self.model = model

        torch.save(model.state_dict(), os.path.join(savedic, "pretain_model.pt"))
        plot(train_loss=train_eloss, test_loss=None, epoch=epochcounter, name=os.path.join(savedic, "pretain"), )
        ndf = pd.DataFrame()
        ndf['train_loss'] = train_eloss
        ndf.to_csv(os.path.join(savedic, "logging.csv"))
        original = original.cuda()
        reconstrustion = model(original.unsqueeze(0).permute(0, 2, 1))
        reconstrustion = reconstrustion.permute(0, 2, 1)
        plot_reconstruction(original, reconstrustion.squeeze(0), savedic)
        # plot(train_loss=train_eloss, test_loss=test_eloss, epoch=epochcounter, name="pretain")
        torch.save(mask_log, os.path.join(savedic, "mask_log"))
        return model

    def data_augment(self,train,train_label,augment_size,augmented_save_path,masking_ratio=0.3):
        newdata=None
        newdata_label=None

        train = torch.tensor(train, dtype=torch.float32)

        train_label = torch.tensor(train_label, dtype=torch.long)
        size=train_label.shape[0]
        count=1
        for sample ,label in zip(train,train_label):
            print("preparing augmented :",count,"/",size)
            count+=1
            model=self.model
            batch= sample.unsqueeze(0).permute(0, 2, 1)
            for _ in range(augment_size):
                if newdata_label is not None:
                    newdata_label=torch.cat((newdata_label,label.unsqueeze(0)))
                else:
                    newdata_label=label.unsqueeze(0)
                masks = noise_mask(batch[0], masking_ratio=masking_ratio)
                masks = torch.from_numpy(masks)
                if self.cuda:
                    batch= batch.cuda()
                    masks = masks.cuda()

                batch.masked_fill_(~masks, 0)

                embedding, batch_result = model(batch)
                if newdata is not None:
                    newdata = torch.cat((newdata,batch_result.permute(0, 2, 1)))
                else:
                    newdata=batch_result.permute(0, 2, 1)
        torch.save(newdata, f=os.path.join(augmented_save_path, "augmented_Data"))
        torch.save(newdata_label, f=os.path.join(augmented_save_path, "augmented_Label"))
        newdata= newdata.cpu().detach().numpy()

        newdata_label = newdata_label.detach().numpy()

        return newdata,newdata_label



    def fit_maskcls(self, model, train, trainlabel, test, test_label, epoch, biglr, smalllr, savepath, verbose=False):

        # dicindex, dicdata, newtrain, newtrainlabels = tool.traindataprepare(train, trainlabel, dic_size=3)

        # dic_data = torch.tensor(dicdata, dtype=torch.float32).permute(0, 2, 1)

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

        model = maskcls(model=model, time_dim=self.param["time_dim"], hidden_dim=self.param["hidden_dim"],
                        classnum=self.param["classnums"])

        criterion = NoFussCrossEntropyLoss()
        ce = nn.CrossEntropyLoss()
        # celoss=NoFussCrossEntropyLoss()
        # coloss=
        linear_param = model.output_layer.parameters()
        ignore = list(map(id, linear_param))
        model_param = filter(lambda p: id(p) not in ignore, model.parameters())
        optimizer = RAdam([{'params': model_param}, {'params': linear_param, 'lr': smalllr}], lr=biglr)
        i = 0  # counting epoch
        savedic = os.path.join(savepath, "cls_model")
        tool.mkdir(savedic)
        espath = os.path.join(savedic, "deepcls.pt")

        es = EarlyStopping(self.earlystopping, path=espath)

        train_eloss = []
        test_eloss = []
        train_eacc = []
        test_eacc = []
        best_test_acc = 0
        xx = []
        if self.cuda:
            model = model.cuda()
            # dic_data = dic_data.cuda(self.gpu)
            test = test.cuda(self.gpu)
            test_label = test_label.cuda(self.gpu)
        while i < epoch:
            if verbose:
                print('Epoch: ', i + 1)
            xx.append(i + 1)
            train_loss = 0
            train_acc = 0
            c = 0
            model = model.train()

            # dic_embed, useless = model(dic_data)
            # newdf = pd.DataFrame(columns=['dic_embedding', 'dic_labels'])
            # newdf['dic_embedding']=dic_embed.tolist()
            # newdf['dic_labels']=

            for batch, (data, target) in enumerate(train_generator):

                if self.cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)

                optimizer.zero_grad()

                embedding, batch_result = model(data)

                #
                # target=target.unsqueeze(1)
                # loss=celoss(batch_result, target)+coloss(embedding,dic_embed)

                # loss = criterion(batch_result, target)
                loss = ce(batch_result, target)

                train_loss = train_loss + loss.item()
                train_acc += acc_count(batch_result, target)

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
            # for batch, (data, target) in enumerate(test_generator):
            #
            #     if self.cuda:
            #         data = data.cuda(self.gpu)
            #         target = target.cuda(self.gpu)
            #
            #     optimizer.zero_grad()
            #
            #     embedding, batch_result = model(data)
            #     # target=target.unsqueeze(1)
            #
            #     loss = criterion(batch_result, target)
            #
            #     test_loss = test_loss + loss.item()
            #     test_acc += acc_count(batch_result, target)
            #
            #     c += 1

            c = 1
            embedding, test_result = model(test)
            test_loss = criterion(test_result, test_label)
            test_acc = acc_count(test_result, test_label)

            print("validation_loss: ", test_loss / c)
            print("validation_acc: ", test_acc / c)
            test_eloss.append(test_loss / c)
            test_eacc.append(test_acc / c)
            es.__call__(test_loss / c, model)
            if test_acc / c > best_test_acc:
                best_test_acc = test_acc / c
                torch.save(model.state_dict(), os.path.join(savedic, "bestacc_checkpoint.pt"))

            if es.early_stops():
                print("Early stopping")
                # 结束模型训练
                break

            i += 1
        print("best_acc:", best_test_acc, "epoch:", i)
        ndf = pd.DataFrame()
        ndf["train_loss"] = train_eloss
        ndf["train_acc"] = train_eacc
        ndf["test_loss"] = test_eloss
        ndf["test_acc"] = test_eacc

        model.load_state_dict(torch.load(espath))
        plot(train_loss=train_eloss, test_loss=test_eloss, epoch=xx, name="fcls_loss")
        plot(train_loss=train_eacc, test_loss=test_eacc, epoch=xx, name="fcls_acc")
        model = model.cpu()
        result = model(test)
        result = result.argmax(dim=-1)
        result = result.cpu().detach().numpy()
        s = metrics.classification_report(test_label, result, output_dict=True)
        print(metrics.classification_report(test_label, result))
        df = pd.DataFrame.from_dict(dict(s)).transpose()
        df.to_csv(os.path.join(savedic, 'result.csv'))
        ndf.to_csv(os.path.join(savedic, 'logging.csv'))

        return model, s['accuracy']

    def fit_maskcls_coce(self, model, train, trainlabel, test, test_label, epoch, biglr, smalllr, savepath,early_stop,samples_weight,
                         verbose=False,data_augment=False,augmented_data=None,augmented_label=None,valid=False):
        dic_size = 3
        print(train.shape)
        print(test.shape)
        validation=None
        validation_label=None
        if valid:
            train,validation,trainlabel,validation_label=train_test_split(train,trainlabel,test_size=0.25,stratify=trainlabel,random_state=414)
        alpha = self.param['alpha']

        train = torch.tensor(train, dtype=torch.float32)  # (all,feature_dim,time_dim)
        test = torch.tensor(test, dtype=torch.float32)

        trainlabel = torch.tensor(trainlabel, dtype=torch.long)
        if data_augment:
            train = torch.cat((train,augmented_data))
            trainlabel = torch.cat((trainlabel,augmented_label))

        test_label = torch.tensor(test_label, dtype=torch.long)
        if valid:
            validation = torch.tensor(validation, dtype=torch.float32)
            validation_label = torch.tensor(validation_label, dtype=torch.long)
            validation = validation.permute(0, 2, 1)
        train = train.permute(0, 2, 1)  # (all,time_dim,feature_dim)
        test = test.permute(0, 2, 1)


        # dicindex, dicdata, newtrain, newtrainlabels = tool.traindataprepare(train, trainlabel, dic_size=dic_size)

        # oldtrain = train
        # oldtrainlabel = trainlabel

        traindata = Data.TensorDataset(train, trainlabel)
        testdata = Data.TensorDataset(test, test_label)

        # sampler = WeightedRandomSampler(samples_weight, samples_num)
        # sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        # train_generator = Data.DataLoader(
        #     traindata, batch_size=self.batch_size, shuffle=False, num_workers=0,sampler=sampler
        # )
        #disable sampler
        train_generator = Data.DataLoader(
            traindata, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        # train_generator = tool.batchcreator(newtrain, newtrainlabels, 3)

        test_generator = Data.DataLoader(
            testdata, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        model = maskcls(model=model, time_dim=self.param["time_dim"], hidden_dim=self.param["hidden_dim"],
                        classnum=self.param["classnums"])

        # using float16
        # model =model.half()

        # criterion = NoFussCrossEntropyLoss()
        celoss = nn.CrossEntropyLoss()

        # coloss = InfoNCE()
        coloss = Contrastiveloss(temp=self.param['temp'])
        linear_param = model.output_layer.parameters()
        ignore = list(map(id, linear_param))
        model_param = filter(lambda p: id(p) not in ignore, model.parameters())
        optimizer = RAdam([{'params': model_param}, {'params': linear_param, 'lr': smalllr}], lr=biglr,weight_decay=self.param["l2_w"])
        i = 0  # counting epoch

        savedic = os.path.join(savepath, "deepcls_coce_model")
        tool.mkdir(savedic)

        espath = os.path.join(savedic, "deepcls_coce.pt")
        es = EarlyStopping(early_stop, path=espath)

        train_eloss = []

        train_ecoloss = []

        test_eloss = []
        train_eacc = []
        test_eacc = []
        val_eloss = None
        val_eacc = None
        best_test_acc = 0
        best_epoch = 0
        xx = []
        if valid:
            val_eloss = []
            val_eacc = []
        if self.cuda:
            model = model.cuda()
        while i < epoch:
            if verbose:
                print('Epoch: ', i + 1)
            xx.append(i + 1)
            train_loss = 0
            train_co_loss = 0
            train_acc = 0
            c = 0
            model = model.train()

            # newdf = pd.DataFrame(columns=['dic_embedding', 'dic_labels'])
            # newdf['dic_embedding']=dic_embed.tolist()
            # newdf['dic_labels']=

            for batch, (data, target) in enumerate(train_generator):
                # print("labels:",target.shape[0]-torch.sum(target),":",torch.sum(target))
                if self.cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)

                optimizer.zero_grad()

                batch_result = model(data)


                CEloss = celoss(batch_result, target)

                Loss = CEloss + self.param["l2_reg"] * l2_reg_loss(model)
                #


                train_loss = train_loss + CEloss.item()

                train_acc += acc_count(batch_result, target)

                c += 1

                Loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=global_norm)
                optimizer.step()

            print("train_loss:", train_loss / c)
            print("train_acc:",train_acc / c)

            train_eloss.append(train_loss / c)

            train_eacc.append(train_acc / c)

            # test_loss = 0
            # test_acc = 0
            # c = 0
            model = model.eval()

            # for batch, (data, target) in enumerate(test_generator):
            #
            #     if self.cuda:
            #         data = data.cuda(self.gpu)
            #         target = target.cuda(self.gpu)
            #
            #     optimizer.zero_grad()
            #
            #     embedding, batch_result = model(data)
            #     # target=target.unsqueeze(1)
            #
            #     loss = celoss(batch_result, target)
            #
            #     test_loss = test_loss + loss.item()
            #     test_acc += acc_count(batch_result, target)
            #
            #     c += 1
            #
            # print("validation_loss: ", test_loss / c)
            # print("validation_acc: ", test_acc / c)
            if valid:
                val_loss, val_acc = self.test(validation, validation_label, model, optimizer, celoss)
                val_eloss.append(val_loss)
                val_eacc.append(val_acc)
            # test_eloss.append(test_loss / c)
            # test_eacc.append(test_acc / c)
            # nni.report_intermediate_result(test_acc / c)
            test_loss,test_acc = self.test(test, test_label, model, optimizer, celoss,val=False)
            test_eloss.append(test_loss)
            test_eacc.append(test_acc)
            # es.__call__(test_loss / c), model)
            es.__call__(train_loss / c, model)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = i
                torch.save(model.state_dict(), os.path.join(savedic, "bestacc_checkpoint.pt"))

            if es.early_stops():
                print("Early stopping")
                # 结束模型训练
                break

            i += 1
        print("best_acc:", best_test_acc, "epoch:", best_epoch)
        model.load_state_dict(torch.load(espath))
        torch.save(model.state_dict(), os.path.join(savedic, "deepcls_coce.pt"))
        plot(train_loss=train_eloss, test_loss=test_eloss,val_loss=val_eloss, epoch=xx, name=os.path.join(savedic, "fcls_loss"))
        plot(train_loss=train_eacc, test_loss=test_eacc,val_loss=val_eacc, epoch=xx, name=os.path.join(savedic, "fcls_acc"))
        model = model.cpu()
        result = model(test)
        result = result.argmax(dim=-1)
        result = result.cpu().detach().numpy()
        s = metrics.classification_report(test_label, result, output_dict=True)
        print(metrics.classification_report(test_label, result))
        ndf = pd.DataFrame()
        ndf["train_loss"] = train_eloss
        ndf["train_acc"] = train_eacc
        ndf["test_loss"] = test_eloss
        ndf["test_acc"] = test_eacc
        ndf["val_loss"] = val_eloss
        ndf["val_acc"] = val_eacc
        best_path = os.path.join(savedic, 'bestacc_' + str(best_test_acc) + '_bestepoch_' + str(best_epoch )+'.txt')
        file = open(best_path, 'w+')
        file.close()
        ndf.to_csv(os.path.join(savedic, 'supervised_logging.csv'))
        df = pd.DataFrame.from_dict(dict(s)).transpose()
        df.to_csv(os.path.join(savedic, 'result.csv'))

        return model, s['accuracy']

    def test_batch(self, test_generator, model, optimizer, celoss):
        test_loss = 0
        test_acc = 0
        c = 0
        for batch, (data, target) in enumerate(test_generator):

            if self.cuda:
                data = data.cuda(self.gpu)
                target = target.cuda(self.gpu)

            optimizer.zero_grad()

            embedding, batch_result = model(data)
            current_loss = celoss(batch_result, target)

            test_loss = test_loss + current_loss.item()
            test_acc += acc_count(batch_result, target)

            c += 1

        print("validation_loss: ", test_loss / c)
        print("validation_acc: ", test_acc / c)
        return test_loss / c, test_acc / c

    def test(self, test, test_label, model, optimizer, celoss,val=True):

        if self.cuda:
            test = test.cuda(self.gpu)
            test_label = test_label.cuda(self.gpu)
        optimizer.zero_grad()

        with torch.no_grad():
            batch_result = model(test)

        current_loss = celoss(batch_result, test_label)
        test_loss = current_loss.item()
        test_acc = acc_count(batch_result, test_label)
        if val:
            print("validation_loss: ", test_loss)
            print("validation_acc: ", test_acc)
        else:
            print("test_loss: ", test_loss)
            print("test_acc: ", test_acc)
        return test_loss, test_acc

    def run_baselines(self,model, train, trainlabel, test, test_label, epoch,lr,savepath,early_stop,samples_weight,
                         verbose=False,data_augment=False,augmented_data=None,augmented_label=None,valid=False):
        print(train.shape)
        print(test.shape)
        validation = None
        validation_label = None
        if valid:
            train, validation, trainlabel, validation_label = train_test_split(train, trainlabel, test_size=0.25,
                                                                               stratify=trainlabel, random_state=414)

        train = torch.tensor(train, dtype=torch.float32)  # (all,feature_dim,time_dim)
        test = torch.tensor(test, dtype=torch.float32)

        trainlabel = torch.tensor(trainlabel, dtype=torch.long)
        if data_augment:
            train = torch.cat((train, augmented_data))
            trainlabel = torch.cat((trainlabel, augmented_label))

        test_label = torch.tensor(test_label, dtype=torch.long)
        if valid:
            validation = torch.tensor(validation, dtype=torch.float32)
            validation_label = torch.tensor(validation_label, dtype=torch.long)
            validation = validation.permute(0, 2, 1)
        train = train.permute(0, 2, 1)  # (all,time_dim,feature_dim)
        test = test.permute(0, 2, 1)

        traindata = Data.TensorDataset(train, trainlabel)
        testdata = Data.TensorDataset(test, test_label)

        # sampler = WeightedRandomSampler(samples_weight, samples_num)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_generator = Data.DataLoader(
            traindata, batch_size=self.batch_size, shuffle=False, num_workers=0, sampler=sampler
        )

        # train_generator = tool.batchcreator(newtrain, newtrainlabels, 3)

        test_generator = Data.DataLoader(
            testdata, batch_size=self.batch_size, shuffle=True, num_workers=0
        )


        # criterion = NoFussCrossEntropyLoss()
        celoss = nn.CrossEntropyLoss()

        optimizer = RAdam(model.parameters(),lr=lr)
        i = 0  # counting epoch

        savedic = os.path.join(savepath, "deepcls_coce_model")
        tool.mkdir(savedic)

        espath = os.path.join(savedic, "deepcls_coce.pt")
        es = EarlyStopping(early_stop, path=espath)

        train_eloss = []

        train_ecoloss = []

        test_eloss = []
        train_eacc = []
        test_eacc = []
        val_eloss = None
        val_eacc = None
        best_test_acc = 0
        best_epoch = 0
        xx = []
        if valid:
            val_eloss = []
            val_eacc = []
        if self.cuda:
            model = model.cuda()
        while i < epoch:
            if verbose:
                print('Epoch: ', i + 1)
            xx.append(i + 1)
            train_loss = 0
            train_co_loss = 0
            train_acc = 0
            c = 0
            model = model.train()


            for batch, (data, target) in enumerate(train_generator):
                # print("labels:",target.shape[0]-torch.sum(target),":",torch.sum(target))
                if self.cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)

                optimizer.zero_grad()
                batch_result = model(data)

                CEloss = celoss(batch_result, target)

                Loss = CEloss + self.param["l2_reg"] * l2_reg_loss(model)

                train_loss = train_loss + CEloss.item()

                train_acc += acc_count(batch_result, target)

                c += 1

                Loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=global_norm)
                optimizer.step()

            print("train_loss:", train_loss / c)
            print("train_acc:", train_acc / c)
            print("train_contrastive_loss", train_co_loss / c)
            train_eloss.append(train_loss / c)
            train_ecoloss.append(train_co_loss / c)
            train_eacc.append(train_acc / c)


            model = model.eval()

            if valid:
                val_loss, val_acc = self.test(validation, validation_label, model, optimizer, celoss)
                val_eloss.append(val_loss)
                val_eacc.append(val_acc)

            test_loss, test_acc = self.test(test, test_label, model, optimizer, celoss, val=False)
            test_eloss.append(test_loss)
            test_eacc.append(test_acc)
            # es.__call__(test_loss / c), model)
            es.__call__(train_loss / c, model)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = i
                torch.save(model.state_dict(), os.path.join(savedic, "bestacc_checkpoint.pt"))

            if es.early_stops():
                print("Early stopping")
                # 结束模型训练
                break

            i += 1
        print("best_acc:", best_test_acc, "epoch:", best_epoch)
        model.load_state_dict(torch.load(espath))
        torch.save(model.state_dict(), os.path.join(savedic, "deepcls_coce.pt"))
        plot(train_loss=train_eloss, test_loss=test_eloss, val_loss=val_eloss, epoch=xx,
             name=os.path.join(savedic, "fcls_loss"))
        plot(train_loss=train_eacc, test_loss=test_eacc, val_loss=val_eacc, epoch=xx,
             name=os.path.join(savedic, "fcls_acc"))
        plot(train_loss=train_ecoloss, test_loss=None, epoch=xx, name=os.path.join(savedic, "fcls_coloss"))
        model = model.cpu()
        result = model(test)
        result = result.argmax(dim=-1)
        result = result.cpu().detach().numpy()
        s = metrics.classification_report(test_label, result, output_dict=True)
        print(metrics.classification_report(test_label, result))
        ndf = pd.DataFrame()
        ndf["train_loss"] = train_eloss
        ndf["train_acc"] = train_eacc
        ndf["test_loss"] = test_eloss
        ndf["test_acc"] = test_eacc
        ndf["val_loss"] = val_eloss
        ndf["val_acc"] = val_eacc
        best_path = os.path.join(savedic, 'bestacc_' + str(best_test_acc) + '_bestepoch_' + str(best_epoch) + '.txt')
        file = open(best_path, 'w+')
        file.close()
        ndf.to_csv(os.path.join(savedic, 'supervised_logging.csv'))
        df = pd.DataFrame.from_dict(dict(s)).transpose()
        df.to_csv(os.path.join(savedic, 'result.csv'))

        return model, s['accuracy']
