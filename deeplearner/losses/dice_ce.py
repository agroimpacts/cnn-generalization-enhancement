import torch
from torch import nn
from .dice import DiceLoss

class DiceCELoss(nn.Module):
    '''
        Combination of dice loss and cross entropy loss through summation

        Params:

            loss_weight (tensor): a manual rescaling weight given to each class. If given, has to be a Tensor of size C
            dice_weight (float): Weight on dice loss for the summation, while weight on cross entropy loss is
                (1 - dice_weight)
            dice_smooth (float): A float number to smooth dice loss, and avoid NaN error, default: 1
            dice_p (int): Denominator value: \sum{x^p} + \sum{y^p}, default: 2
            ignore_index (int): Class index to ignore

        Returns:

            Loss tensor

    '''

    def __init__(self, loss_weight = None, dice_weight=0.5 , dice_smooth=1, dice_p=1, ignore_index=-100):
        super(DiceCELoss, self).__init__()
        self.loss_weight = loss_weight
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth
        self.dice_p = dice_p
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size do not match"

        dice = DiceLoss(weight=self.loss_weight, ignore_index=self.ignore_index, smooth=self.dice_smooth, p=self.dice_p)
        ce = nn.CrossEntropyLoss(weight=self.loss_weight, ignore_index=self.ignore_index)
        loss = self.dice_weight * dice(predict, target) + (1 - self.dice_weight) * ce(predict, target)

        return loss


class BalancedDiceCELoss(nn.Module):
    '''
        Dice Cross Entropy weighted by inverse of label frequency

        Params:

            ignore_index (int): Class index to ignore
            predict (torch.tensor): Predicted tensor of shape [N, C, *]
            target (torch.tensor): Target tensor either in shape [N,*] or of same shape with predict
            other args pass to DiceCELoss, excluding loss_weight

        Returns:

            Same as DiceCELoss

    '''

    def __init__(self, ignore_index=-100, **kwargs):
        super(BalancedDiceCELoss, self).__init__()
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

        loss = DiceCELoss(loss_weight=lossWeight, **self.kwargs)

        return loss(predict, target)
