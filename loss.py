import random
import re

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from torch.nn import functional as F


class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor):
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask

        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)


        return self.mse_loss(masked_pred, masked_true)

def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():

        if re.search("output_layer",name):
            return torch.sum(torch.square(param))




def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]




class metricloss(nn.Module):
    def __init__(self, temp):
        super(metricloss, self).__init__()
        self.temp = temp

    def forward(self, embedding, label):
        """calculate the contrastive loss
            """
        # cosine similarity between embeddings
        cosine_sim = cosine_similarity(embedding, embedding)
        cosine_sim = cosine_sim / self.temp
        cosine_sim = np.exp(cosine_sim)


        # calculate outer sum
        contrastive_loss = 0
        for i in range(len(embedding)):
            n_i = label.tolist().count(label[i]) - 1
            inner_sum = 0
            # calculate inner sum
            for j in range(len(embedding)):
                if label[i] == label[j] and i != j:
                    inner_sum = inner_sum + cosine_sim[i][j]
            if n_i != 0:
                contrastive_loss += (inner_sum / n_i)
            else:
                contrastive_loss += 0
        return contrastive_loss

class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
