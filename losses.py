import torch.nn as nn

__all__ = [
    "MixedLoss"
]


class MixedLoss(nn.Module):
    def __init__(self,
                 class_loss_weight: float = 0.5,
                 regression_loss_weight: float = 1.0,
                 split_classification_loss_weight: float = 0.5):
        super().__init__()

        self.classification_loss_weight = class_loss_weight
        self.regression_loss_weight = regression_loss_weight
        self.split_classification_loss_weight = split_classification_loss_weight

        self.classification_ce = nn.CrossEntropyLoss()
        self.split_classification_ce = nn.CrossEntropyLoss()
        self.regression_mse = nn.L1Loss()

    def forward(self, y_pred, y_true):
        classification_pred, regression_pred, split_cls_pred = y_pred
        classification_true, regression_true, split_cls_true = y_true

        classification_loss = self.classification_ce(
            classification_pred, classification_true
        ) * self.classification_loss_weight

        regression_loss = self.regression_mse(
            regression_pred.squeeze(), regression_true
        ) * self.regression_loss_weight

        split_classification_loss = self.split_classification_ce(
            split_cls_pred, split_cls_true
        ) * self.split_classification_loss_weight

        return classification_loss + regression_loss + split_classification_loss
