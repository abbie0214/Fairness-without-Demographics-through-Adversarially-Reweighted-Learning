# training

import torch
from torch import nn
import config
from model import Primary_NN, Adversary_NN
from loss import primary_loss, adversary_loss
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning import metrics
from pytorch_lightning.metrics import Accuracy, Recall, Precision, ConfusionMatrix

from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import numpy as np
from fairness_matrics import protected_group_split, fairness_matrics, fairness_output

import config
args = config.parse_args()

from collections import defaultdict
class Test():
    def __init__(self, dataloader_test,  Primary_NN, Adversary_NN):
        super().__init__()


        self.primary_net = Primary_NN
        self.adversary_net = Adversary_NN

        self.primary_net.eval()
        self.adversary_net.eval()

        self.loss_func_1= primary_loss()
        self.loss_func_2 = adversary_loss()

        self.dataloader_test = dataloader_test

        self.epoch_loss = []
        self.epoch_acc = []

        self.epoch_FP_s0 = []

    def tester(self, epoch):

        self.tot_losses = 0
        self.tot_losses_1 = 0
        self.tot_losses_2 = 0


        self.total = 0
        self.num_correct_test = 0

        self.num_correct_test_s0 = 0
        self.num_correct_test_s1 = 0


        for self.step, (self.batch_x, self.batch_y,self.batch_s) in enumerate(self.dataloader_test):
            

            self.gt =self.batch_y

            self.pose_weights = torch.ones((self.batch_y.shape[0], 1))

            # self.z, self.logits, self.prob = self.primary_net(self.batch_x.float())
            self.z, self.logits, self.pred = self.primary_net(self.batch_x.float())

            self.example_weights = self.adversary_net(self.batch_x.float())

            # self.batch_y = torch.squeeze(self.batch_y)

            # self.batch_y = self.batch_y.long()


            self.loss_1 = self.loss_func_1(self.batch_y,self.logits,self.example_weights)

            self.loss_2 = self.loss_func_2(self.batch_y, self.logits, self.example_weights,pos_weights=self.pose_weights, adversary_loss_type='ce_loss')

            self.tot_losses_1 += self.loss_1.item() * self.batch_x.shape[0]
            self.tot_losses_2 += self.loss_2.item() * self.batch_x.shape[0]

            # self.tot_losses += self.loss.item() * self.batch_x.shape[0]

            # _, self.pred = self.prob.max(1)

            # print('self.prob',self.prob)
            self.total += self.batch_y.size(0)
            self.num_correct_test += self.pred.eq(self.batch_y).sum().item()
            self.test_acc = self.num_correct_test / self.total

            # print('self.gt',self.gt)
            # print('self.s',self.batch_s)


            protected_group_otps = protected_group_split(self.pred,self.gt, self.batch_s)
            fn_metrics_r0,fn_metrics_r1,fn_metrics_s0,fn_metrics_s1,fn_metrics_0,fn_metrics_1,fn_metrics_2,fn_metrics_3 = fairness_output(protected_group_otps,epoch)
        #########################################################################################

        ########################################################################################

        self.epoch_acc.append(self.test_acc)
        self.epoch_loss.append( self.tot_losses_1 / (len(self.dataloader_test)*5267))


        print('self.test_acc',self.test_acc)
        print('test loss',self.tot_losses_1 / (len(self.dataloader_test) * 5267))
        # print('self.tot_losses_1',self.tot_losses_1 / (len(self.dataloader_test)*32))

        test_output = OrderedDict({
            # 'tot_loss': self.tot_losses / (len(self.dataloader) * 32),
            'classification_loss': self.tot_losses_1 / (len(self.dataloader_test) * 5267),
            'adversary_loss': self.tot_losses_2 / (len(self.dataloader_test) * 5267),
            'acc': self.test_acc,

            # $METRIC  takes value in [accuracy, recall, precision, -- perf metrics
            #           tp, tn, fp, fn, --confusion matrix
            #           fpr, -- false positive  rate
            #           tpr, -- true positive  rate
            #           tnr, -- true negative  rate
            #           ]

        })

        return test_output