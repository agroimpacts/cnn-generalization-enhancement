from torch.autograd import Variable
from .losses import *

def validate(valData, model, criterion, buffer, valLoss, gpu):
    """
        Validate model

        Params:
            valData (''DataLoader''): Batch grouped data
            model: Trained model for validation
            criterion: Function to calculate loss
            buffer: Buffer added to the targeted grid when creating dataset. This allows loss to calculate
                at non-buffered region
            valLoss (empty list): To record average loss for each epoch
            gpu (binary,optional): Decide whether to use GPU, default is True

    """

    model.eval()

    # mini batch iteration
    epoch_loss = 0
    i = 0

    for img, label in valData:

        img = Variable(img, requires_grad=False)
        label = Variable(label, requires_grad=False)

        # GPU setting
        if gpu:
            img = img.cuda()
            label = label.cuda()

        out = model(img)

        loss = criterion(out[:, :, buffer:-buffer, buffer:-buffer],
                         label[:, buffer:-buffer, buffer:-buffer])
        epoch_loss += loss.item()
        i += 1

    print('validation loss: {}'.format(epoch_loss / i))

    if valLoss != None:
        valLoss.append(float(epoch_loss / i))
