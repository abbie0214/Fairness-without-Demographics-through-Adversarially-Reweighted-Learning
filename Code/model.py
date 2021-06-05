import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import config
args = config.parse_args()

primary_hidden_units  = [64, 32]
adversary_hidden_units=[32]

class Primary_NN(nn.Module):
    def __init__(self,indim, primary_hidden_units):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(indim, primary_hidden_units[0]),
            nn.ReLU(),
            nn.Linear(primary_hidden_units[0],  primary_hidden_units[1]),
            nn.ReLU(),
        )
        # self.classlayer = nn.Linear(primary_hidden_units[1],2)
        # self.sigmoidlayer = nn.Softmax(dim = 1)
        ######
        self.classlayer =  nn.Linear(primary_hidden_units[1],1)
        self.sigmoidlayer = nn.Sigmoid()

    def forward(self, x):
        z = self.model(x)
        logits = self.classlayer(z)
        # sigmoid_output = self.sigmoidlayer(logits)
        # prob = torch.max(sigmoid_output,torch.tensor(0.5))
        # return z, logits, sigmoid_output   # previously is prob not sigmoid_output
        #####
        sigmoid_output = self.sigmoidlayer(logits)
        # print('sigmoid_output',sigmoid_output)
        sigmoid_output[sigmoid_output > 0.5] = 1
        sigmoid_output[sigmoid_output <= 0.5] = 0
        # print('sigmoid_output',sigmoid_output)
        return z, logits, sigmoid_output



class Adversary_NN(nn.Module):
    def __init__(self,indim,adversary_hidden_units):

        super().__init__()
        self.model  = nn.Sequential(
            nn.Linear(indim,adversary_hidden_units[0]),
            nn.Linear(adversary_hidden_units[0],1,bias=True),
        )

    def forward(self, x):
        """Applies sigmoid to adversary output layer and returns normalized example weight."""
        # adv_output_layer = self.model(x)
        # example_weights = torch.sigmoid(adv_output_layer)
        #
        # mean_example_weights = torch.mean(example_weights)
        # example_weights /= torch.maximum(mean_example_weights, torch.tensor(1e-4))
        #
        #
        # example_weights = torch.ones_like(example_weights) + example_weights
        adv_output_layer = self.model(x)
        out_1 = torch.sigmoid(adv_output_layer)

        mean_example_weights = torch.mean(out_1)

        out_2 = out_1/torch.maximum(mean_example_weights, torch.tensor(1e-4))


        example_weights = torch.ones_like(out_2) + out_2


        return example_weights



