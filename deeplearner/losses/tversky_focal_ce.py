import torch
from torch import nn
from .tversky_focal import TverskyFocalLoss

class TverskyFocalCELoss(nn.Module):
    '''
        Combination of tversky focal loss and cross entropy loss though summation

        Params:

            loss_weight (tensor): a manual rescaling weight given to each class. If given, has to be a Tensor of size C
            tversky_weight (float): Weight on tversky focal loss for the summation, while weight on cross entropy loss
            is (1 - tversky_weight)
            tversky_smooth (float): A float number to smooth tversky focal loss, and avoid NaN error, default: 1
            tversky_alpha (float):
            tversky_gamma (float):
            ignore_index (int): Class index to ignore

        Returns:

            Loss tensor

    '''

    def __init__(self, loss_weight=None, tversky_weight=0.5, tversky_smooth=1, tversky_alpha=0.7, tversky_gamma=1.33, ignore_index=-100):
        super(TverskyFocalCELoss, self).__init__()
        self.loss_weight = loss_weight
        self.tversky_weight = tversky_weight
        self.tversky_smooth = tversky_smooth
        self.tversky_alpha = tversky_alpha
        self.tversky_gamma = tversky_gamma
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size do not match"

        tversky = TverskyFocalLoss(weight=self.loss_weight, ignore_index=self.ignore_index, smooth=self.tversky_smooth,
                                   alpha=self.tversky_alpha, gamma=self.tversky_gamma)
        ce = nn.CrossEntropyLoss(weight=self.loss_weight, ignore_index=self.ignore_index)
        loss = self.tversky_weight * tversky(predict, target) + (1 - self.tversky_weight) * ce(predict, target)

        return loss


class LocallyWeightedTverskyFocalCELoss(nn.Module):
    '''
        Combination of tversky focal loss and cross entropy loss weighted by inverse of label frequency

        Params:
            
            ignore_index (int): Class index to ignore
            predict (torch.tensor): Predicted tensor of shape [N, C, *]
            target (torch.tensor): Target tensor either in shape [N,*] or of same shape with predict
            other args pass to DiceCELoss, excluding loss_weight

        Returns:

            Same as TverskyFocalCELoss

    '''

    def __init__(self, ignore_index=-100, **kwargs):
        super(LocallyWeightedTverskyFocalCELoss, self).__init__()
        self.ignore_index =  ignore_index
        self.kwargs = kwargs

    def forward(self, predict, target):
        # get class weights
        unique, unique_counts = torch.unique(target, return_counts=True)
        # calculate weight for only valid indices
        unique_counts = unique_counts[unique != self.ignore_index]
        unique = unique[unique != self.ignore_index]
        ratio = unique_counts.float() / torch.numel(target)
        weight = (1. / ratio) / torch.sum(1. / ratio)

        lossWeight = torch.ones(predict.shape[1]).cuda() * 0.00001
        for i in range(len(unique)):
            lossWeight[unique[i]] = weight[i]

        loss = TverskyFocalCELoss(loss_weight=lossWeight, **self.kwargs)

        return loss(predict, target)
