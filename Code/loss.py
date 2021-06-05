import torch
from torch import nn


class primary_loss(nn.Module):   # compute loss
    def __init__(self):
        super().__init__()

        self.ce_loss = nn.BCEWithLogitsLoss()

    def forward(self,class_labels, logits, example_weights):

        loss = self.ce_loss(logits,class_labels)
        # print('ce loss',loss)
        primary_weighted_loss = (example_weights * loss)
        primary_weighted_loss = torch.mean(primary_weighted_loss)
        # print('primary_weighted_loss',primary_weighted_loss)
        return primary_weighted_loss


class adversary_loss(nn.Module):  #compute weight
    def __init__(self):
        super().__init__()

        self.hinge_loss = nn.HingeEmbeddingLoss()
        self.ce_loss = nn.BCEWithLogitsLoss()

    def forward(self,class_labels, logits, example_weights,pos_weights,adversary_loss_type):
        if adversary_loss_type == 'hinge_loss':
            loss = self.hinge_loss(logits,class_labels,pos_weights)


        # pos_weights: a float tensor of shape [batch_size, 1]. Assigns weight 1
        #       for positive examples, and weight 0 for negative examples in the batch.

        elif adversary_loss_type == 'ce_loss':
            loss = self.ce_loss(logits,class_labels)

        # print('loss.shape',loss.shape)
        # print('example_weights.shape',example_weights.shape)
        adversary_weighted_loss = - 1 * (example_weights * loss)



        # print('adversary_weighted_loss',adversary_weighted_loss)
        # print('torch.mean(adversary_weighted_loss).shape',torch.mean(adversary_weighted_loss))
        return torch.mean(adversary_weighted_loss)


