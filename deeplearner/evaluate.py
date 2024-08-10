import pandas as pd

from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from .metrics import BinaryMetrics


def evaluate(evalData='', model='', buffer='', gpu='', bucket=None, outPrefix='', filename=None):
    """
    Evaluate model
    Params:
        evalData (''DataLoader''): Batch grouped data
        model: Trained model for validation
        buffer: Buffer added to the targeted grid when creating dataset. This allows metrics to calculate only
            at non-buffered region
        gpu (binary,optional): Decide whether to use GPU, default is True
        bucket (str): name of s3 bucket to save metrics
        outPrefix (str): s3 prefix to save metrics
    """

    model.eval()

    metrics = []

    for img, label in evalData:
        img = Variable(img, requires_grad=False)
        label = Variable(label, requires_grad=False)

        # GPU setting
        if gpu:
            img = img.cuda()
            label = label.cuda()
        out = model(img)

        # Compute metrics
        out = F.softmax(out, 1)
        batch, nclass, height, width = out.size()

        for i in range(batch):
            label_batch = label[i, buffer:-buffer, buffer:-buffer].cpu().numpy()
            batch_predict = out.max(dim=1)[1][:, buffer:-buffer, buffer:-buffer].data[i].cpu().numpy()
            for n in range(1, nclass):
                class_out = out[:, n, buffer:-buffer, buffer:-buffer].data[i].cpu().numpy()
                class_predict = np.where(batch_predict == n, 1, 0)
                class_label = np.where(label_batch == n, 1, 0)
                metrics_chip = BinaryMetrics(class_label, class_out, class_predict)
                # append if exists
                try:
                    metrics[n - 1].append(metrics_chip)
                except:
                    metrics.append([metrics_chip])

    metrics = [sum(m) for m in metrics]
    report = pd.DataFrame({
        'tss': [m.tss() for m in metrics],
        'accuracy': [m.accuracy() for m in metrics],
        'precision': [m.precision() for m in metrics],
        'recall': [m.recall() for m in metrics],
        'fpr': [m.false_positive_rate() for m in metrics],
        'F1-score': [m.F1_measure() for m in metrics],
        'IoU': [m.iou() for m in metrics],
        'AUC': [m.area_under_roc() for m in metrics]
    }, index=["class_{}".format(m) for m in range(1, len(metrics) + 1)])

    if not filename:
        filename = "Metrics"
    dir_metrics = f"{outPrefix}/{filename}.csv"
    #dir_metrics = f"s3://{bucket}/{outPrefix}/{filename}.csv"
    report.to_csv(dir_metrics)
