import torch
# import torch.nn as nn
import torch.nn.functional as F
from load_data_real import Custom_dataset
from load_segmentation import Custom_dataset as Custom_dataset_Seg
from torch.utils.data import DataLoader
import numpy as np
from classifier import classifier
from attention_model import attention
from MI_estimator import estimator
import collections as cls
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
import multiprocessing as mp
import networkx as nx
import os
import itertools as its
from torch.optim import lr_scheduler

HOME = os.environ['HOME']





class my_model():

    def __init__(self, dynamic='HR', name=None,  noise_std=0.2, seed=12, num=10.,
                 threshold=0.6 ):

        self.batch_size_a = 32
        self.batch_size_c = 10

        self.epoch = 400  # 1000
        self.lr_a = 1e-4  # self.lr*(1/self.lammda)
        self.lr_c = 1e-3
        self.dyn_dim = 1

        self.dynamic = dynamic
        self.name = name
        self.noise_std = noise_std
        self.seed = seed
        self.num = num  #number of train sample
        self.threshold = threshold

        self.past = 3
        self.future = 1

        self.path = HOME + f'/cause/model_saver/real/{dynamic}/{name}/coupling_att/noise_std={noise_std}/seed={seed}/num={num}/'

        if not os.path.exists(self.path):
            os.makedirs(self.path)


    def optimizer(self, model, model_name, lr):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)  # , weight_decay=5e-4 )
        checkpoint_dir = self.path + '/{0}.pth'.format(model_name)
        return optimizer, checkpoint_dir

    def C_loss( self, logit, label ):
        return F.binary_cross_entropy_with_logits( logit, label )


    def KL_loss_a( self, logit_joint, logit_indep, att ):  # , label ):

        logit_joint = torch.mean(logit_joint * att, dim=-1)
        logit_indep = torch.log( torch.mean(torch.exp(logit_indep * att), dim=-1) + 1e-12 )
        KL = logit_joint - logit_indep  # eq(12)
        return KL
        # return logit_joint, logit_indep


    def KL_loss( self, logit_joint, logit_indep, att ):  # , label ):
        #
        # negative_samples =  int( logit_joint.shape[-1] * att.mean() * 0.1 )
        # # index = torch.argsort( att, dim=-1 )
        # index = torch.randperm( logit_joint.shape[-1] )
        #
        # logit_joint_sort = logit_joint[ index ]
        # logit_joint_ns = logit_joint_sort[ :negative_samples ]  #negative_samples
        # logit_joint = torch.cat( [logit_joint * att, logit_joint_ns], dim=-1 )
        # logit_joint = torch.mean( logit_joint, dim=-1 )

        # logit_indep_sort = logit_indep[ index ]
        # logit_indep_ns = logit_indep_sort[ :negative_samples ]  # negative_samples
        # logit_indep = torch.cat( [logit_indep * att, logit_indep_ns], dim=-1 )
        # logit_indep = torch.log( torch.mean(torch.exp(logit_indep), dim=-1) + 1e-12 )

        logit_joint = torch.mean(logit_joint * att, dim=-1)
        logit_indep = torch.log(torch.mean(torch.exp(logit_indep* att), dim=-1) + 1e-12)

        KL = logit_joint - logit_indep  # eq(12)
        return KL
        # return logit_joint, logit_indep


    def load_model(self, model, optimizer, checkpoint_dir):
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['epoch']
        return step

    def save_model(self, model, optimizer, epoch, checkpoint_dir):
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, checkpoint_dir)


    def train(self,load):

        train_dataset = Custom_dataset(self.dynamic,  name=self.name, stage='train', seed=self.seed,
                                       noise_std=self.noise_std,  purpose = 'ATEn',
                                       num=self.num, past=self.past, future=self.future, dyn_dim=self.dyn_dim,
                                        )  # seed是抽样percent训练边的概率种子
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size_a, shuffle=True, num_workers=1,
                                  persistent_workers=True)  # , pin_memory=True )#, drop_last=True )

        train_dataset_c = Custom_dataset(self.dynamic,  name=self.name, stage='train_c', seed=self.seed,
                                       noise_std=self.noise_std,  purpose = 'classify',
                                       num=self.num, past=self.past, future=self.future, dyn_dim=self.dyn_dim,
                                        )
        train_loader_c = DataLoader(dataset=train_dataset_c, batch_size=self.batch_size_c, shuffle=True, num_workers=1,
                                    persistent_workers=True)  # , pin_memory=True )#, drop_last=True )


        valid_dataset = Custom_dataset(self.dynamic,  name=self.name, stage='valid', seed=self.seed,
                                       noise_std=self.noise_std,  purpose = 'classify',
                                       num=self.num, past=self.past, future=self.future, dyn_dim=self.dyn_dim,
                                        )
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size_c, num_workers=1,
                                  persistent_workers=True)  # , pin_memory=True )#, drop_last=True )


        test_dataset = Custom_dataset(self.dynamic,  name=self.name, stage='test', seed=self.seed,
                                       noise_std=self.noise_std,  purpose = 'classify',
                                       num=self.num, past=self.past, future=self.future, dyn_dim=self.dyn_dim,
                                        )
        test_loader = DataLoader( dataset=test_dataset, batch_size=self.batch_size_c, num_workers=1,
                                  persistent_workers=True )  # , pin_memory=True )#, drop_last=True )


        self.a = attention(self.dyn_dim).cuda()
        self.d = estimator(in_dim=self.past*self.dyn_dim * 2 + self.future*self.dyn_dim).cuda()
        self.d2 = estimator(in_dim=self.past*self.dyn_dim  + self.future*self.dyn_dim ).cuda()

        op_a, cd_a = self.optimizer(self.a, 'attention', self.lr_a)
        scheduler_a = lr_scheduler.ExponentialLR(op_a, gamma=0.999)
        op_d, cd_d = self.optimizer(self.d, 'mi_estimator1', self.lr_a)
        scheduler_d = lr_scheduler.ExponentialLR(op_d, gamma=0.999)
        op_d2, cd_d2 = self.optimizer(self.d2, 'mi_estimator2', self.lr_a)
        scheduler_d2 = lr_scheduler.ExponentialLR(op_d2, gamma=0.999)

        # self.c = classifier(in_dim=self.dyn_dim).cuda()
        # op_c, cd_c = self.optimizer(self.c, 'classifier2', self.lr_c)
        # scheduler_c = lr_scheduler.ExponentialLR(op_c, gamma=0.999)


        if load:
            step = self.load_model(self.a, op_a, cd_a)
            _ = self.load_model( self.d, op_d,  cd_d)
            _ = self.load_model( self.d2, op_d,  cd_d2)
            print( 'load_step:{0}'.format(step) )

            auc_ATE, auprc_ATE = self.classify( test_loader )
            # auc_ATE, auprc_ATE = self.ATEn( d, d2 )
            return auc_ATE, auprc_ATE

        else:
            step = 1



        auc_valid_best = 0
        for i in range(step, self.epoch + 1):
            TE, CF, ATT = [], [], []

            for sample, joint, indep, joint2, indep2, label in train_loader:  # sample [bs,2]
                sample = sample.to('cuda')  # [bs,ch,2,l]
                joint = joint.to('cuda')  # [bs,l,2*past+feature]
                indep = indep.to('cuda')  # [bs,l,2*past+feature]
                joint2 = joint2.to('cuda')  # [bs,l,,past+feature]
                indep2 = indep2.to('cuda')  # [bs,l,past+feature]
                label = label.to('cuda')

                op_d.zero_grad(), op_d2.zero_grad()
                with torch.no_grad():
                    att = self.a(sample)  # [bs, l]
                cp = torch.cat([joint, indep], dim=0)
                logit_cp = self.d(cp)  # [2bs,l]
                # print( logit_cp.shape )
                logit_cp_joint, logit_cp_indep = logit_cp[:joint.shape[0]], logit_cp[-joint.shape[
                    0]:]  # [bs,l],[bs,le]
                sf = torch.cat([joint2, indep2], dim=0)
                logit_sf = self.d2( sf )
                logit_sf_joint, logit_sf_indep = logit_sf[:joint2.shape[0]], logit_sf[
                                                                             -joint2.shape[0]:]  # [bs,l]
                # print( logit_cp_joint.shape, logit_cp_indep.shape )
                indep_cp = (logit_cp_indep < 70)
                indep_sf = (logit_sf_indep < 70)
                logit_cp_joint = logit_cp_joint[indep_cp]  # [bs,length] -- [bs*length]
                logit_cp_indep = logit_cp_indep[indep_cp ]  # [bs,length] -- [bs*length]
                logit_sf_joint = logit_sf_joint[indep_sf ]
                logit_sf_indep = logit_sf_indep[indep_sf ]
                # print( logit_cp_joint.shape, logit_cp_indep.shape )

                # KL_cp = self.KL_loss(logit_cp_joint, logit_cp_indep, att[:joint.shape[0]][indep_cp])  # , label_batch )
                # KL_sf = self.KL_loss(logit_sf_joint, logit_sf_indep, att[:joint.shape[0]][indep_sf])  # , label_batch )
                KL_cp = self.KL_loss(logit_cp_joint, logit_cp_indep, att[indep_cp])  # , label_batch )
                KL_sf = self.KL_loss(logit_sf_joint, logit_sf_indep, att[indep_sf])  # , label_batch )

                KL_cp = KL_cp[ torch.isfinite( KL_cp ) ]
                KL_sf = KL_sf[ torch.isfinite(KL_sf) ]

                loss = -(torch.mean(KL_cp) + torch.mean(KL_sf))  #

                loss.backward()
                op_d.step(), op_d2.step()

                #train attention
                op_a.zero_grad()#
                att = self.a(sample)  # [bs, length ]

                with torch.no_grad():
                    logit_cp = self.d(cp)  # [2bs,l]
                    logit_sf = self.d2(sf)

                logit_cp_joint, logit_cp_indep = logit_cp[:joint.shape[0]], logit_cp[-joint.shape[0]:]
                logit_sf_joint, logit_sf_indep = logit_sf[:joint2.shape[0]], logit_sf[
                                                                             -joint2.shape[0]:]  # [bs,length-10]

                indep = (logit_cp_indep < 70) & (logit_sf_indep < 70)
                logit_cp_joint = logit_cp_joint[indep]
                logit_cp_indep = logit_cp_indep[indep]
                logit_sf_joint = logit_sf_joint[indep]
                logit_sf_indep = logit_sf_indep[indep]
                KL_cp = self.KL_loss_a(logit_cp_joint, logit_cp_indep, att[indep])
                KL_sf = self.KL_loss_a(logit_sf_joint, logit_sf_indep, att[indep])

                finite = torch.isfinite(KL_cp) & torch.isfinite(KL_sf)
                KL_cp = KL_cp[finite]
                KL_sf = KL_sf[finite]

                te = torch.mean(KL_cp - KL_sf)  # transfer entropy

                loss = -te
                loss.backward( )
                op_a.step()


                TE.append(te.cpu().detach().numpy())
                ATT.append( torch.mean(att).cpu().detach().numpy()  )





            scheduler_a.step(), scheduler_d.step(), scheduler_d2.step() #, scheduler_c.step()


            if i % 1 == 0 or i == self.epoch:  # or ATT<0.1:

                TE = np.mean(TE)
                ATT = np.mean(ATT)

                for _ in range(5):

                    self.c = classifier(in_dim=self.dyn_dim).cuda()
                    op_c, cd_c = self.optimizer(self.c, 'classifier2', self.lr_c)
                    # scheduler_c = lr_scheduler.ExponentialLR(op_c, gamma=0.999)

                    for _ in range(1):
                        CF = []
                        pred, Label = [], []
                        for sample, label in train_loader_c:  # sample [bs,2]
                            sample = sample.to('cuda')  # [bs,ch,2,l]
                            label = label.to('cuda')

                            op_c.zero_grad()
                            # op_a.zero_grad()
                            with torch.no_grad():
                                att = self.a(sample)  # [bs, l]
                            att[att < self.threshold] = 0
                            logit = self.c.forward2( sample, att )
                            cf = self.C_loss( logit, label )  # #ce loss

                            loss = cf  #
                            loss.backward()
                            op_c.step()
                            # op_a.step()
                            #
                            CF.append(cf.cpu().detach().numpy())
                            pred.extend( torch.sigmoid(logit).detach().cpu().numpy() )
                            Label.extend( label.detach().cpu().numpy() )

                        # scheduler_c.step()

                    # CF = np.mean(CF)
                    # auc = roc_auc_score(np.array(Label), np.array(pred))
                    # auprc = average_precision_score(np.array(Label), np.array(pred))


                    print(
                        f'dynamic:{self.dynamic} name={self.name} noise_std:{self.noise_std} '
                        f'seed:{self.seed} train epoch:{i} TE:{TE} att:{ATT} '
                        # f'CF:{CF} train_auc:{auc} train_auprc:{auprc}'
                    )

                    # if i % 20 == 0:

                    auc_valid , _ = self.classify( valid_loader )

                    if auc_valid >= auc_valid_best:

                        auc_test, auprc_test = self.classify( test_loader )

                        auc_valid_best = auc_valid
                        epoch_best = i
                        self.save_model(self.c, op_c, i, cd_c)
                        self.save_model(self.a, op_a, i, cd_a)
                        self.save_model(self.d, op_d, i, cd_d)
                        self.save_model(self.d2, op_d, i, cd_d2)

                    print(f'dynamic:{self.dynamic} name:{self.name} noise_std:{self.noise_std}'
                          f' valid_auc:{auc_valid} valid_auc_best:{auc_valid_best} '
                          f'test_auc:{auc_test}, test_auprc:{auprc_test} epoch_best:{epoch_best}')



        #train the classifier further
        print( 'train the classifier further' )
        _ = self.load_model( self.a, op_a, cd_a)
        _ = self.load_model( self.c, op_c, cd_c )

        auc_valid, _ = self.classify(valid_loader)
        print(f'auc_valid:{auc_valid} auc_valid_best:{auc_valid_best}')

        for i in range( 50 ):
            for sample, label in train_loader_c:  # sample [bs,2]
                sample = sample.to('cuda')  # [bs,ch,2,l]
                label = label.to('cuda')

                op_c.zero_grad()
                # op_a.zero_grad()
                with torch.no_grad():
                    att = self.a(sample)  # [bs, l]
                att[att < self.threshold] = 0
                logit = self.c.forward2(sample, att)
                cf = self.C_loss(logit, label)  # #ce loss

                loss = cf  #
                loss.backward()
                op_c.step()

            auc_valid, _ = self.classify(valid_loader)
            # print( f'auc_valid:{auc_valid} auc_valid_best:{auc_valid_best}' )

            if auc_valid >= auc_valid_best:
                auc_test, auprc_test = self.classify(test_loader)

                auc_valid_best = auc_valid
                self.save_model(self.c, op_c, i, cd_c)

                print(f'auc_valid_best:{auc_valid_best}')

        return auc_test, auprc_test


    def classify( self , dataset_loader ):

        pred_test, Label_test = [], []
        for sample, label in dataset_loader:  # sample [bs,2]
            sample = sample.to('cuda')  # [bs,2,indexs]
            # print( sample_batch.shape )
            with torch.no_grad():
                att = self.a(sample)
                att[att < self.threshold] = 0
                logit = self.c.forward2(sample, att)

            pred_test.extend(torch.sigmoid(logit).detach().cpu().numpy())
            Label_test.extend(label.numpy())
        Label_test, pred_test = np.array(Label_test), np.array(pred_test)
        # print(Label_test, pred_test)
        auc_test = roc_auc_score(Label_test, pred_test)
        auprc_test = average_precision_score(Label_test, pred_test)


        return auc_test, auprc_test






def run(gpu_num=0, dynamic='HR', name=None, load=False, noise_std=0.1, seed=12, num=10):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

    model = my_model( dynamic=dynamic, name=name, noise_std=noise_std, seed=seed, num=num )
    print( 'dynamic:{0} noise_std:{1} seed:{2} num:{3}'.format(dynamic, noise_std, seed, num) )

    auc_test, auprc_test = model.train( load= False )

    return auc_test, auprc_test



if __name__ == '__main__':

    num=10


    # for name in [ 'cat', 'macaque', 'mouse', 'celegans', 'rat' ]:  # 'mouse',
    # for name in ['cat']:
    # for name in [ 'celegans']:
    # for name in ['mouse']:
    # for name in ['rat']:
    for name in ['macaque']:

        AUC = cls.defaultdict(dict)
        # for dyn in ['HR']:
        # for dyn in ['Izh']:
        # for dyn in ['Rulkov']:
        # for dyn in ['FHN']:
        # for dyn in ['Morris']:

        for dyn in ['HR', 'Izh', 'Rulkov', 'FHN']:

            if name == 'mouse' and dyn=='FHN':
                continue

            AUC[name][dyn] = []

            for seed in range(10):

                auc, auprc = run( name=name,  gpu_num=0, dynamic=dyn, noise_std=0.1,
                                  load=False, seed=seed, num=num )

                AUC[name][dyn].append([auc, auprc])

                print(AUC)
