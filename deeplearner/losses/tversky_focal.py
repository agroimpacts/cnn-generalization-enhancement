import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryTverskyFocalLoss(nn.Module):
    '''

    Pytorch versiono of tversky focal loss proposed in paper
    'A novel focal Tversky loss function and improved Attention U-Net for lesion segmentation'
    (https://arxiv.org/abs/1810.07842)

    Params:

        smooth (float): A float number to smooth loss, and avoid NaN error, default: 1
        alpha (float): Hyperparameters alpha, paired with (1 - alpha) to shift emphasis to improve recall
        gamma (float): Tversky index, default: 1.33
        predict (torch.tensor): Predicted tensor of shape [N, C, *]
        target (torch.tensor): Target tensor either in shape [N,*] or of same shape with predict


    Returns:

        Loss tensor

    '''

    def __init__(self, smooth=1, alpha=0.7, gamma=1.33):
        super(BinaryTverskyFocalLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = 1 - self.alpha
        self.gamma = gamma


    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size do not match"

        # no reduction, same as original paper
        predict = predict.contiguous().view(-1)
        target = target.contiguous().view( -1)

        num = (predict * target).sum() + self.smooth
        den = (predict * target).sum() + self.alpha * ((1 - predict) * target).sum() \
              + self.beta * (predict * (1 - target)).sum() + self.smooth
        loss = torch.pow(1 - num/den, 1 / self.gamma)

        return loss


class TverskyFocalLoss(nn.Module):
    '''

    Tversky focal loss

    Params:

        weight (torch.tensor): Weight array of shape [num_classes,]
        ignore_index (int): Class index to ignore
        predict (torch.tensor): Predicted tensor of shape [N, C, *]
        target (torch.tensor): Target tensor either in shape [N,*] or of same shape with predict
        other args pass to BinaryTverskyFocalLoss

    Returns:

        same as BinaryTverskyFocalLoss

    '''
    def __init__(self, weight=None, ignore_index=-100, **kwargs):
        super(TverskyFocalLoss, self).__init__()
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

        tversky = BinaryTverskyFocalLoss(**self.kwargs)
        total_loss = 0
        if self.weight is None:
            self.weight = torch.Tensor([1. / nclass] * nclass).cuda()
        else:
            if isinstance(self.weight, list):
                self.weight = torch.tensor(self.weight, dtype=torch.float64).cuda()
        
        #weight = torch.Tensor([1./nclass] * nclass).cuda() if self.weight is None else self.weight
        predict = F.softmax(predict, dim=1)
        
        for i in range(nclass):
            if i != self.ignore_index:
                tversky_loss = tversky(predict[:, i], target[:, i])
                assert self.weight.shape[0] == nclass, \
                    'Expect weight shape [{}], get[{}]'.format(nclass, self.weight.shape[0])
                tversky_loss *= self.weight[i]
                total_loss += tversky_loss
            
        return total_loss


class LocallyWeightedTverskyFocalLoss(nn.Module):
    '''

        Tversky focal loss weighted by inverse of label frequency

        Params:

            ignore_index (int): Class index to ignore
            predict (torch.tensor): Predicted tensor of shape [N, C, *]
            target (torch.tensor): Target tensor either in shape [N,*] or of same shape with predict
            other args pass to BinaryTverskyFocalLoss

        Returns:

            same as TverskyFocalLoss

    '''

    def __init__(self, ignore_index=-100, **kwargs):
        super(LocallyWeightedTverskyFocalLoss, self).__init__()
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

        # loss
        loss = TverskyFocalLoss(weight=lossWeight, ignore_index=self.ignore_index, **self.kwargs)

        return loss(predict, target)
