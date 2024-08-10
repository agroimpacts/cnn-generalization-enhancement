import numpy as np
import pandas as pd
from sklearn import metrics

from .tools import InputError

class BinaryMetrics:
    '''
    Metrics measuring model performance.
    '''

    def __init__(self, refArray, scoreArray, predArray=None):
        '''
        Params:
            refArray (narray): Array of ground truth
            scoreArray (narray): Array of pixels scores of positive class
            predArray (narray): Boolean array of predictions telling whether a pixel belongs to a specific class

        '''

        self.eps = 10e-6
        self.observation = refArray.flatten()
        self.score = scoreArray.flatten()
        if predArray is not None:
            self.prediction = predArray.flatten()
        # take score over 0.5 as prediction if predArray not provided
        else:
            self.prediction = np.where(self.score > 0.5, 1, 0)
        self.confusion_matrix = self.confusion_matrix()

        if self.observation.shape != self.score.shape:
            raise InputError("Inconsistent input shape")

    def __add__(self, other):
        """
        Add two BinaryMetrics instances
        Params:
            other (''BinaryMetrics''): A BinaryMetrics instance
        Return:
            ''BinaryMetrics''
        """

        return BinaryMetrics(np.append(self.observation, other.observation),
                             np.append(self.score, other.score),
                            np.append(self.prediction, other.prediction))


    def __radd__(self, other):
        """
        Add a BinaryMetrics instance with reversed operands
        Params:
            other
        Returns:
            ''BinaryMetrics
        """

        if other == 0:
            return self
        else:
            return self.__add__(other)


    def confusion_matrix(self):
        """
        Calculate confusion matrix of given ground truth and predicted label
        Returns:
            ''pandas.dataframe'' of observation on the column and prediction on the row
        """

        refArray = self.observation
        predArray = self.prediction

        if refArray.max() > 1 or predArray.max() > 1:
            raise Exception("Invalid array")
        predArray = predArray * 2
        sub = refArray - predArray

        self.tp = np.sum(sub == -1)
        self.fp = np.sum(sub == -2)
        self.fn = np.sum(sub == 1)
        self.tn = np.sum(sub == 0)

        confusionMatrix = pd.DataFrame(data=np.array([[self.tn, self.fp], [self.fn, self.tp]]),
                                       index=['observation = 0', 'observation = 1'],
                                       columns = ['prediction = 0', 'prediction = 1'])
        return confusionMatrix


    def iou(self):
        """
        Calculate interception over union
        Returns:
            float
        """

        return metrics.jaccard_score(self.observation, self.prediction)


    def precision(self):
        """
        Calculate precision
        Returns:
            float
        """

        return metrics.precision_score(self.observation, self.prediction)


    def recall(self):
        """
        Calculate recall
        Returns:
            float
        """

        return metrics.recall_score(self.observation, self.prediction)


    def accuracy(self):
        """
        Calculate accuracy
        Returns:
            float
        """

        return metrics.accuracy_score(self.observation, self.prediction)


    def tss(self):
        """
        Calculate true scale statistic (TSS)
        Returns:
            float
        """

        return self.tp / (self.tp + self.fn) + self.tn / (self.tn + self.fp) - 1


    def false_positive_rate(self):
        """
        Calculate false positive rate
        Returns:
             float
        """

        return self.fp / (self.tn + self.fp)

    def F1_measure(self):
        """
        Calculate F1 score.
        Returns:
            float
        """

        try:
            precision = self.tp / (self.tp + self.fp)
            recall = self.tp / (self.tp + self.fn)
            f1 = (2 * precision * recall) / (precision + recall)

        except ZeroDivisionError:
            precision = self.tp / (self.tp + self.fp + self.eps)
            recall = self.tp / (self.tp + self.fn + self.eps)
            f1 = (2 * precision * recall) / (precision + recall + self.eps)

        return f1


    def area_under_roc(self):
        """
        Compute Area Under the Curve (AUC)
        Returns:
            float
        """

        return metrics.roc_auc_score(self.observation, self.score)