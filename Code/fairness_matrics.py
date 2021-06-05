#fairness matrics

import torch
from torch import nn
import config
from model import Primary_NN, Adversary_NN
from loss import primary_loss,adversary_loss
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning import metrics
from pytorch_lightning.metrics import Recall,Precision,ConfusionMatrix
import torchmetrics

import numpy as np

from torchmetrics.functional import accuracy

import config
args = config.parse_args()
import math
# Compute metrics follow this naming convention:
#         -- metrics computed on the entire data:
#             "label $METRIC"
#         -- metrics computed for each sensitive group in the data:
#             "GROUPNAMEsensitiveclassCLASS_NUMBER $METRIC"
#         -- metrics computed at a number of thresholds (given N_THRESHOLD):
#             "label $METRIC"
#             "GROUPNAMEsensitiveclassCLASS_NUMBER $METRIC th"


# protected groups = ['sex'],['race']
#subgroups = ['female white ', 'male black ', 'male white ', 'female black ' ]    GT label + (sex + race)

def protected_group_split(pred,gt,s):
    #gt + sex + race
    # sex : male 0 , female 1
    #race: white 0 , black 1,       10 11 00 01

    # pred = torch.unsqueeze(pred, dim=1)


    label = torch.cat((pred, gt, s), 1)  # [0,1,2,3]  pred + gt + sex + race

    # print('label',label)

    sex_0 = label[label[:,2] == 0]
    sex_1 = label[label[:,2] == 1]

    race_0 = label[label[:,3] == 0]
    race_1 = label[label[:,3] == 1]


    subgroup_0 = label[[a and b for a,b in zip(label[:,2] == 1,label[:,3] == 0)]]
    subgroup_1 = label[[a and b for a,b in zip(label[:,2] == 1,label[:,3] == 1)]]
    subgroup_2 = label[[a and b for a,b in zip(label[:,2] == 0,label[:,3] == 0)]]
    subgroup_3 = label[[a and b for a,b in zip(label[:,2] == 0,label[:,3] == 1)]]

    all = label
    protected_group_otps = OrderedDict({
        'sex_0':sex_0,
        'sex_1':sex_1,
        'race_0':race_0,
        'race_1':race_1,
        'subgroup_0': subgroup_0,
        'subgroup_1': subgroup_1,
        'subgroup_2': subgroup_2,
        'subgroup_3': subgroup_3,
        'all':all,
    })


    return protected_group_otps

def fairness_matrics(group):
    # $METRIC  takes value in [accuracy, recall, precision, -- perf metrics
    #           tp, tn, fp, fn, --confusion matrix
    #           fpr, -- false positive  rate
    #           tpr, -- true positive  rate
    #           tnr, -- true negative  rate
    #           ]
    pred = group[:,0]

    gt = group[:,1].long()

    if gt.shape > torch.Size([1]) :
        confusion_matrix = pl.metrics.ConfusionMatrix(num_classes=2)
        confmax = confusion_matrix(pred, gt)
        acc = torchmetrics.functional.accuracy(pred,gt)
        # print('gt.shape',gt.shape)
        # print('confmax',confmax)

        TN = confmax[0,0]
        FP = confmax[0,1]
        FN = confmax[1,0]
        TP = confmax[1,1]
        FPR = FP / (TN + FP + 1e-5)
        FNR = FN / (FN + TP + 1e-5)
        Recall = TP / (TP + FP+ 1e-5)
        Precision = TP / (TP + FN+ 1e-5)



        # fn_metrics = OrderedDict({
        #     'TN':TN,

        #     'FP ':FP ,
        #     'FN':FN,
        #     'TP':TP,
        #     ' FPR': FPR ,
        #     'FNR ': FNR,
        #     'Recall ': Recall,
        #     'Precision ': Precision,
        #     'acc':acc
        # })
        #
        # print('TN',TN)
        # print('FP',FP)
        # print('FN',FN)
        # print('TP',TP)
        # print('FPR',FPR)
        # print('FNR',FNR)
        # print('Recall',Recall)
        # print('Precision',Precision)
        # print('acc',acc)

        # print('auc_{}'.format(group), fn_metrics)
    # print(' Precision',Precision)  # pred + gt + sex + race
        if Recall != 0:

            auroc = pl.metrics.classification.AUROC(num_classes=2,pos_label=1)
            auc = auroc(pred,gt)

            #
            # print('auc_{}'.format(group),auc)


            # return fn_metrics,auc

            fn_metrics = OrderedDict({
                # 'TN': TN,
                'FP': FP,
                'FN': FN,
                # 'TP': TP,
                'FPR': FPR,
                'FNR': FNR,
                'Recall': Recall,
                'Precision': Precision,
                'acc': acc,
                'auc':auc,
            })

            # print('TN', TN)
            # print('FP', FP)
            # print('FN', FN)
            # print('TP', TP)
            # print('FPR', FPR)
            # print('FNR', FNR)
            # print('acc', acc)
            # print('auc',auc)

            return fn_metrics


def fairness_output(protected_group_otps,epoch):

    sex_0 = protected_group_otps['sex_0']
    sex_1 = protected_group_otps['sex_1']

    race_0 = protected_group_otps['race_0']
    race_1 = protected_group_otps['race_1']

    subgroup_0 = protected_group_otps['subgroup_0']
    subgroup_1 = protected_group_otps['subgroup_1']
    subgroup_2 = protected_group_otps['subgroup_2']
    subgroup_3 = protected_group_otps['subgroup_3']

    all = protected_group_otps['all']

    fn_metrics_s0 = fairness_matrics(sex_0)
    fn_metrics_s1= fairness_matrics(sex_1)

    fn_metrics_r0 = fairness_matrics(race_0)
    fn_metrics_r1 = fairness_matrics(race_1)

    fn_metrics_0= fairness_matrics(subgroup_0)
    fn_metrics_1 = fairness_matrics(subgroup_1)
    fn_metrics_2 = fairness_matrics(subgroup_2)
    fn_metrics_3 = fairness_matrics(subgroup_3)

    fn_metrics = fairness_matrics(all)
    #
    # print('fn_metrics_s0',fn_metrics_s0)
    # print('fn_metrics_s1',fn_metrics_s1)
    #
    # print('fn_metrics_r0',fn_metrics_r0)
    # print('fn_metrics_r1',fn_metrics_r1)
    #
    # print('fn_metrics_0',fn_metrics_0)
    # print('fn_metrics_1',fn_metrics_1)
    # print('fn_metrics_2',fn_metrics_2)
    # print('fn_metrics_3',fn_metrics_3)
    #
    # print('fn_metrics',fn_metrics)
    

    auc_s0 = fn_metrics_s0['auc']
    auc_s1 = fn_metrics_s1['auc']
    auc_r0 = fn_metrics_r0['auc']
    auc_r1 = fn_metrics_r1['auc']

    auc_0 = fn_metrics_0['auc']
    auc_1 = fn_metrics_1['auc']
    auc_2 = fn_metrics_2['auc']
    auc_3 = fn_metrics_3['auc']

    auc = fn_metrics['auc']

    aucs_1 = np.array([auc_s0,auc_s1,auc_r0,auc_r1])
    aucs_2 = np.array([auc_s0,auc_s1,auc_r0,auc_r1,auc_0,auc_1,auc_2,auc_3])
    aucs_3 = np.array([auc_0,auc_1,auc_2,auc_3])


    auc_min_1 = min(aucs_1[np.where(aucs_1 != None)])
    auc_min_2 = min(aucs_2[np.where(aucs_2 != None)])
    auc_min_3 = min(aucs_3[np.where(aucs_3 != None)])

    # print('auc_min_1,auc_min_2,auc_min_3',auc_min_1,auc_min_2,auc_min_3)


    auc_macro_1 = torch.tensor(aucs_1[np.where(aucs_1 != None)].sum().item()/4)
    auc_macro_2 = torch.tensor(aucs_2[np.where(aucs_2 != None)].sum().item()/8)
    auc_macro_3 = torch.tensor(aucs_3[np.where(aucs_3 != None)].sum().item()/4)
    # print('auc_macro_1,auc_macro_2,auc_macro_3',auc_macro_1,auc_macro_2,auc_macro_3)


    min_idx = np.where(min(len(sex_0),len(sex_1),len(race_0),len(race_1)))
    auc_minority_1 = fairness_matrics(protected_group_otps[['sex_0','sex_1','race_0','race_1'][min_idx[0][0]]])['auc']
    # if auc_minority_1 != None:
    #     print('auc_minority_1',auc_minority_1)


    min_idx2 = np.where(min([i for i in [len(sex_0),len(sex_1),len(race_0),len(race_1), len(subgroup_0),len(subgroup_1),len(subgroup_2),len(subgroup_3)] if i > 0] ))
    auc_minority_2 = fairness_matrics(protected_group_otps[['sex_0','sex_1','race_0','race_1','subgroup_0','subgroup_1','subgroup_2','subgroup_3'][min_idx2[0][0]]])['auc']
    # if auc_minority_2 != None:
    #     print('auc_minority_2',auc_minority_2)

    min_idx3 = np.where(min([i for i in [len(subgroup_0),len(subgroup_1),len(subgroup_2),len(subgroup_3)] if i > 0] ))
    auc_minority_3 = fairness_matrics(protected_group_otps[['subgroup_0','subgroup_1','subgroup_2','subgroup_3'][min_idx3[0][0]]])['auc']
    # if auc_minority_3 != None:
        # print('auc_minority_3',auc_minority_3)



    if epoch == args.num_epochs - 1:


        print('race group 0',fn_metrics_r0)
        print('race group 1',fn_metrics_r1)

        print('sex group 0',fn_metrics_s0)
        print('sex group 1',fn_metrics_s1)

        print('subgroup 0',fn_metrics_0)
        print('subgroup 1',fn_metrics_1)
        print('subgroup 2',fn_metrics_2)
        print('subgroup 3',fn_metrics_3)

        print('all',fn_metrics)

        print('auc_min_1,auc_min_2,auc_min_3',auc_min_1,auc_min_2,auc_min_3)
        print('auc_macro_1,auc_macro_2,auc_macro_3',auc_macro_1,auc_macro_2,auc_macro_3)
        print('auc_minority_1,auc_minority_2,auc_minority_3',auc_minority_1,auc_minority_2,auc_minority_3)


    return fn_metrics_r0,fn_metrics_r1,fn_metrics_s0,fn_metrics_s1,fn_metrics_0,fn_metrics_1,fn_metrics_2,fn_metrics_3




################################################################################
    # auc_s0 = fairness_matrics(sex_0)
    # auc_s1 = fairness_matrics(sex_1)
    #
    # auc_r0 = fairness_matrics(race_0)
    # auc_r1 = fairness_matrics(race_1)
    #
    # auc_00 = fairness_matrics(subgroup_00)
    # auc_01 = fairness_matrics(subgroup_01)
    # auc_10 = fairness_matrics(subgroup_10)
    # auc_11 = fairness_matrics(subgroup_11)
    #
    # #
    # # print('auc_s0',auc_s0)
    # # print('auc_s1',auc_s1)
    # # print('auc_r0',auc_r0)
    # # print('auc_r1',auc_r1)
    # #
    # # print('auc_00',auc_00)
    # # print('auc_01',auc_01)
    # # print('auc_10',auc_10)
    # # print('auc_11',auc_11)
    #
    # aucs_1 = np.array([auc_s0,auc_s1,auc_r0,auc_r1])
    # aucs_2 = np.array([auc_s0,auc_s1,auc_r0,auc_r1,auc_00,auc_01,auc_10,auc_11])
    # aucs_3 = np.array([auc_00,auc_01,auc_10,auc_11])
    #
    #
    # auc_min_1 = min(aucs_1[np.where(aucs_1 != None)])
    # auc_min_2 = min(aucs_2[np.where(aucs_2 != None)])
    # auc_min_3 = min(aucs_3[np.where(aucs_3 != None)])
    # # print('auc_min_1',auc_min_1)
    # print('auc_min_1,auc_min_2,auc_min_3',auc_min_1,auc_min_2,auc_min_3)
    #
    #
    # auc_macro_1 = torch.tensor(aucs_1[np.where(aucs_1 != None)].sum().item()/4)
    # auc_macro_2 = torch.tensor(aucs_2[np.where(aucs_2 != None)].sum().item()/8)
    # auc_macro_3 = torch.tensor(aucs_3[np.where(aucs_3 != None)].sum().item()/4)
    # print('auc_macro_1,auc_macro_2,auc_macro_3',auc_macro_1,auc_macro_2,auc_macro_3)
    # # print('auc_macro_1',auc_macro_1)
    #
    # min_idx = np.where(min(len(sex_0),len(sex_1),len(race_0),len(race_1)))
    # auc_minority_1 = fairness_matrics(protected_group_otps[['sex_0','sex_1','race_0','race_1'][min_idx[0][0]]])
    # if auc_minority_1 != None:
    #     print('auc_minority_1',auc_minority_1)
    #
    #
    # min_idx2 = np.where(min([i for i in [len(sex_0),len(sex_1),len(race_0),len(race_1), len(subgroup_00),len(subgroup_01),len(subgroup_10),len(subgroup_11)] if i > 0] ))
    # auc_minority_2 = fairness_matrics(protected_group_otps[['sex_0','sex_1','race_0','race_1','subgroup_00','subgroup_01','subgroup_10','subgroup_11'][min_idx2[0][0]]])
    # if auc_minority_2 != None:
    #     print('auc_minority_2',auc_minority_2)
    #
    # min_idx3 = np.where(min([i for i in [len(subgroup_00),len(subgroup_01),len(subgroup_10),len(subgroup_11)] if i > 0] ))
    # auc_minority_3 = fairness_matrics(protected_group_otps[['subgroup_00','subgroup_01','subgroup_10','subgroup_11'][min_idx3[0][0]]])
    # if auc_minority_3 != None:
    #     print('auc_minority_3',auc_minority_3)
    #
    #
    # # auc_min_1 = min(auc_s0,auc_s1,auc_r0,auc_r1)
    # # auc_min_2 = min(auc_s0,auc_s1,auc_r0,auc_r1,auc_00,auc_01,auc_10,auc_11)
    # # auc_min_3 = min(auc_00,auc_01,auc_10,auc_11)
    # # print('auc_min_1,auc_min_2,auc_min_3',auc_min_1,auc_min_2,auc_min_3)
    # #
    # # auc_macro_1 = (auc_s0 + auc_s1 + auc_r0 + auc_r1)/4
    # # auc_macro_2 = (auc_s0 + auc_s1+auc_r0+auc_r1+auc_00+auc_01+auc_10+auc_11)/8
    # # auc_macro_3 = (auc_00+auc_01+auc_10+auc_11)/4
    # # print('auc_macro_1,auc_macro_2,auc_macro_3',auc_macro_1,auc_macro_2,auc_macro_3)
    # #
    # # min_idx = np.where(min(len(sex_0),len(sex_1),len(race_0),len(race_1)))
    # # auc_minority = fairness_matrics(protected_group_otps[['sex_0','sex_1','race_0','race_1'][min_idx[0][0]]])
    # # print('auc_minority',auc_minority)
    # #
    # # min_idx2 = np.where(min(len(sex_0),len(sex_1),len(race_0),len(race_1), len(subgroup_00),len(subgroup_01),len(subgroup_10),len(subgroup_11)))
    # # auc_minority2 = fairness_matrics(protected_group_otps[['sex_0','sex_1','race_0','race_1','subgroup_00','subgroup_01','subgroup_10','subgroup_11'][min_idx2[0][0]]])
    # # print('auc_minority2',auc_minority2)
    # #
    # # min_idx3 = np.where(min(len(subgroup_00),len(subgroup_01),len(subgroup_10),len(subgroup_11)))
    # # auc_minority3 = fairness_matrics(protected_group_otps[['subgroup_00','subgroup_01','subgroup_10','subgroup_11'][min_idx3[0][0]]])
    # # print('auc_minority3',auc_minority3)

################################################################################