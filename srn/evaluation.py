"""Evaluation metrics"""
from ignite.metrics import Metric
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score

class AverageLoss(Metric):
    """
    Calculates average AverageLossloss.
    """
    def __init__(self, output_transform=lambda x: x):
        self.acc_loss = None
        super().__init__(output_transform=output_transform)

    def reset(self):
        self.acc_loss = []

    def update(self, loss):
        """
        Parameters
        ----------
        loss : scalar
            loss value
        """
        self.acc_loss.append(loss)

    def compute(self):
        return np.mean(self.acc_loss)

class F1(Metric):
    """
    Calculates F1 measure.
    """
    def __init__(self, output_transform=lambda x: x):
        self.acc_cnd = None
        self.acc_target = None

        super().__init__(output_transform=output_transform)

    def reset(self):
        self.acc_target = []
        self.acc_cnd = []

    def update(self, output):
        """
        Parameters
        ----------
        target : numpy vector of integer
            ground truth class
        cnd : numpy vector of integer
            confidence class
        """
        target, cnd = output
        # accumulators to calculate f1
        self.acc_target.append(target)
        self.acc_cnd.append(cnd)

    def compute(self):
        """
        Returns
        -------
            recall
            precision
            f1
        """
        if not self.acc_pred or not self.acc_target:
            raise ValueError("No item inserted yet")

        total_cnd = np.concatenate(self.acc_cnd)
        total_target = np.concatenate(self.acc_target)
        total_pred = np.where(total_cnd > 0.5, 1, 0)

        num_samples = total_cnd.shape[0] # pylint: disable=E1136  # pylint/issues/3139
        num_pos = (total_pred == 1).sum()
        ave_num_pos = num_pos / num_samples

        recall = recall_score(y_true=total_target,
                              y_pred=total_pred,
                              average='samples',
                              zero_division=0)
        precision = precision_score(y_true=total_target,
                                    y_pred=total_pred,
                                    average='samples',
                                    zero_division=0)
        f1 = f1_score(y_true=total_target,
                      y_pred=total_pred,
                      average='samples',
                      zero_division=0)

        return recall, precision, f1, ave_num_pos
