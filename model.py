import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

from custom_densenet import densenet121


class NALU(nn.Module):
    def __init__(self, out_features: int, in_features: int, epsilon: float = 1e-6):
        super().__init__()
        self.tanh_weights = Parameter(torch.Tensor(out_features, in_features))
        self.sigmoid_weights = Parameter(torch.Tensor(out_features, in_features))
        self.gate_weights = Parameter(torch.Tensor(1, in_features))
        self.epsilon = Parameter(epsilon, requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.tanh_weights)
        init.xavier_uniform_(self.sigmoid_weights)
        init.xavier_uniform_(self.gate_weights)

    def forward(self, input):
        tanh_weights = torch.tanh(self.tanh_weights)
        sigmoid_weights = torch.sigmoid(self.sigmoid_weights)

        W = tanh_weights * sigmoid_weights
        g = F.sigmoid(F.linear(input, self.gate_weights))

        m = torch.exp(F.linear(torch.log(torch.abs(input) + self.epsilon), W))

        a = F.linear(input, W)
        nalu = g * a + (1 - g) * m
        return nalu


class PretrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        pretrained_model = densenet121(pretrained=True, drop_rate=0.1).train()
        self.extractor = pretrained_model.features
        self.classifier = nn.Sequential(
            nn.Linear(pretrained_model.classifier.in_features, 256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

        self.regressor = nn.Sequential(
            nn.Linear(pretrained_model.classifier.in_features, 512),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        is_test = len(x.size()) == 5
        if is_test:
            b, ncrops, c, h, w = x.size()
            features = self.extractor(x.view(-1, c, h, w))
            out = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
            out = F.dropout(out, p=0.1, training=self.training)

            class_out = self.classifier(out) \
                .view(b, ncrops, -1) \
                .mean(dim=1)
            regression_out = self.regressor(out) \
                .view(b, ncrops, -1) \
                .mean(dim=1)

        else:
            features = self.extractor(x)
            out = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
            out = F.dropout(out, p=0.1, training=self.training)

            class_out = self.classifier(out)
            regression_out = self.regressor(out)

        return class_out, regression_out
