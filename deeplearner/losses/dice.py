import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryDiceLoss(nn.Module):
    '''

        Dice loss of binary class

        Params:

            smooth (float): A float number to smooth loss, and avoid NaN error, default: 1
            p (int): Denominator value: \sum{x^p} + \sum{y^p}, default: 2
            predict (torch.tensor): Predicted tensor of shape [N, *]
            target (torch.tensor): Target tensor of same shape with predict

        Returns:

            Loss tensor

    '''

    def __init__(self, smooth=1, p=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size do not match"
        predict = predict.contiguous().view(-1)
        target = target.contiguous().view(-1)

        num = 2 * (predict * target).sum() + self.smooth
        den = (predict.pow(self.p) + target.pow(self.p)).sum() + self.smooth
        loss = 1 - num / den

        return loss



class DiceLoss(nn.Module):
    '''

        Dice loss

        Params:

            weight (torch.tensor): Weight array of shape [num_classes,]
            ignore_index (int): Class index to ignore
            predict (torch.tensor): Predicted tensor of shape [N, C, *]
            target (torch.tensor): Target tensor either in shape [N,*] or of same shape with predict
            other args pass to BinaryDiceLoss

        Returns:

            same as BinaryDiceLoss

    '''

    def __init__(self, weight=None, ignore_index=-100, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        nclass = predict.shape[1]
        if predict.shape == target.shape:
            pass
        elif len(predict.shape) == 4:
            target = F.one_hot(target, num_classes=nclass).permute(0, 3, 1, 2).contiguous()
        else:
            assert 'predict shape not applicable'

        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        weight = torch.Tensor([1. / nclass] * nclass).cuda() if self.weight is None else self.weight
        predict = F.softmax(predict, dim=1)

        for i in range(nclass):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])

                assert weight.shape[0] == nclass, \
                    'Expect weight shape [{}], get[{}]'.format(nclass, weight.shape[0])
                dice_loss *= weight[i]
                total_loss += dice_loss

        return total_loss


class BalancedDiceLoss(nn.Module):
    '''

        Dice Loss weighted by inverse of label frequency

        Params:

                ignore_index (int): Class index to ignore
                predict (torch.tensor): Predicted tensor of shape [N, C, *]
                target (torch.tensor): Target tensor either in shape [N,*] or of same shape with predict
                other args pass to BinaryDiceLoss

        Returns:

            same as BinaryDiceLoss

    '''

    def __init__(self, ignore_index=-100, **kwargs):
        super(BalancedDiceLoss, self).__init__()
        self.kwargs = kwargs
        self.ignore_index = ignore_index

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

        loss = DiceLoss(weight=lossWeight, ignore_index=self.ignore_index, **self.kwargs)

        return loss(predict, target)
