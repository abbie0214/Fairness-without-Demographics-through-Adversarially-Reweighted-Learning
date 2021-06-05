#main

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm
from training import Train
from testing import Test
from data_loader import data_set
import config
import matplotlib.pyplot as plt

from model import Primary_NN, Adversary_NN



args = config.parse_args()

indim = args.indim

primary_hidden_units  = [64, 32]
adversary_hidden_units=[32]

class Main():
    def __init__(self,dataloader,dataloader_test):

        self.primary_net = Primary_NN(indim = indim, primary_hidden_units = primary_hidden_units)
        self.adversary_net = Adversary_NN(indim = indim,adversary_hidden_units= adversary_hidden_units)

        self.dataloader = dataloader
        self.dataloader_test = dataloader_test

        self.train = Train(self.dataloader,self.primary_net,self.adversary_net)
        self.test= Test( self.dataloader_test,self.primary_net,self.adversary_net)

    def train_model(self,epoch):
        train_output = self.train.trainer(epoch)
        return train_output

    def test_model(self,epoch):
        test_output = self.test.tester(epoch)
        return test_output

    def plott(self,epoch):

        plt.title('train loss')
        plt.plot(range(len(self.train.epoch_loss)), self.train.epoch_loss)
        plt.savefig("results/train_loss.png")
        plt.clf()


        ###
        plt.title('test loss')
        plt.plot(range(len(self.test.epoch_loss)), self.test.epoch_loss)
        plt.savefig("results/test_loss")
        plt.clf()

        plt.title('train acc')
        plt.plot(range(len(self.train.epoch_acc)), self.train.epoch_acc)
        plt.savefig("results/train_acc.png")
        plt.clf()


        ###
        plt.title('test acc')
        plt.plot(range(len(self.test.epoch_acc)), self.test.epoch_acc)
        plt.savefig("results/test_acc.png")
        plt.clf()

dataloader, dataloader_test = data_set()


main_obj = Main(dataloader,dataloader_test)


num_epochs = args.num_epochs
for epoch in tqdm(range(num_epochs)):
    train_output = main_obj.train_model(epoch)

    with torch.no_grad():
        test_output = main_obj.test_model(epoch)

main_obj.plott(num_epochs)
