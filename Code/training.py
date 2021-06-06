#training

import torch
from torch import nn
import config
from model import Primary_NN, Adversary_NN
from loss import primary_loss,adversary_loss
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning import metrics
from pytorch_lightning.metrics import Accuracy, Recall,Precision,ConfusionMatrix

from torch.utils.tensorboard import SummaryWriter
from fairness_matrics import protected_group_split, fairness_matrics, fairness_output
import config
args = config.parse_args()


class Train():
    def __init__(self, dataloader,Primary_NN,Adversary_NN):
        super().__init__()

        self.primary_net = Primary_NN
        self.adversary_net = Adversary_NN

        self.primary_net.train()
        self.adversary_net.train()


        self.loss_func_1= primary_loss()
        self.loss_func_2 = adversary_loss()


        self.optimizer = torch.optim.Adagrad((filter(lambda p: p.requires_grad,self.primary_net.parameters())) ,
                                                 lr = args.lr_primary) # 0.001 works good, 0.01 original param


        self.adversary_optimizer = torch.optim.Adagrad((filter(lambda p: p.requires_grad,self.adversary_net.parameters())) ,
                                                 lr = args.lr_adversary)# 0.001 works good, 0.01 original param

        self.dataloader = dataloader

        self.epoch_loss = []
        self.epoch_acc = []


    def trainer(self, epoch):

        self.tot_losses = 0
        self.tot_losses_1 = 0
        self.tot_losses_2 = 0


        self.total = 0
        self.num_correct_train = 0
        self.num_correct_sens_train = 0

        # writer = SummaryWriter("my_experiment")

        for self.step, (self.batch_x, self.batch_y,self.batch_s) in enumerate(self.dataloader):
            # print('self.batch_y',self.batch_y)
            # print('self.batch_s',self.batch_s)

            self.gt =self.batch_y

            self.pose_weights = torch.ones((self.batch_y.shape[0], 1))

            # self.z, self.logits, self.prob  = self.primary_net(self.batch_x.float())
            self.z, self.logits, self.pred = self.primary_net(self.batch_x.float())

            # print('self.logits',self.logits)
            # print('self.batch_y',self.batch_y)

            self.example_weights = self.adversary_net(self.batch_x.float())

            # self.batch_y = torch.squeeze(self.batch_y)
            # self.batch_y = self.batch_y.long()

            # print('self.logits',self.logits.shape)
            # print('self.batch_y',self.batch_y.shape)
            # print('#########################epoch##########################',epoch)
            self.loss_1 = self.loss_func_1(self.batch_y,self.logits,self.example_weights.detach().clone())
            # print('loss_1',self.loss_1)


            self.optimizer.zero_grad()
            self.adversary_optimizer.zero_grad()
            self.loss_1.backward()
            # print('#########################epoch##########################',epoch)
            # for p in self.primary_net.parameters():
            #     print('primary',p.requires_grad)
            self.tot_losses_1 += self.loss_1.item() * self.batch_x.shape[0]

            if epoch >= 2:
                self.loss_2 = self.loss_func_2(self.batch_y, self.logits.detach().clone(), self.example_weights,
                                               pos_weights=self.pose_weights, adversary_loss_type='ce_loss')
                # print('loss_2',self.loss_2)
                self.loss_2.backward()
                self.tot_losses_2 += self.loss_2.item()* self.batch_x.shape[0]
            self.optimizer.step()
            self.adversary_optimizer.step()
            # print('#########################epoch##########################',epoch)
            # for p in self.adversary_net.parameters():
            #     print('adversary',p.requires_grad)



            # _,self.pred = self.prob.max(1)


            self.total += self.batch_y.size(0)
            self.num_correct_train += self.pred.eq(self.batch_y).sum().item()
            self.train_acc = self.num_correct_train / self.total

            # print('################ training ################')
            # protected_group_otps = protected_group_split(self.pred,self.gt, self.batch_s)
            # fairness_output(protected_group_otps,epoch)

        # print('self.train_acc',self.train_acc)
        # print('self.tot_losses_1',self.tot_losses_1 / (len(self.dataloader)*32))

        self.epoch_acc.append(self.train_acc)
        self.epoch_loss.append( self.tot_losses_1 / (len(self.dataloader)*args.batch_size))

        train_output = OrderedDict({
            'classification_loss': self.tot_losses_1 / (len(self.dataloader)*args.batch_size),
            'adversary_loss': self.tot_losses_2 / (len(self.dataloader)*args.batch_size),
            'acc': self.train_acc,

        # $METRIC  takes value in [accuracy, recall, precision, -- perf metrics
        #           tp, tn, fp, fn, --confusion matrix
        #           fpr, -- false positive  rate
        #           tpr, -- true positive  rate
        #           tnr, -- true negative  rate
        #           ]

        })

        return train_output
