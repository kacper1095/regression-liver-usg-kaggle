import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

from custom_densenet import densenet121


class NAC(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.tanh_weights = Parameter(torch.Tensor(out_features, in_features))
        self.sigmoid_weights = Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.tanh_weights)
        init.xavier_uniform_(self.sigmoid_weights)

    def forward(self, input):
        tanh_weights = torch.tanh(self.tanh_weights)
        sigmoid_weights = torch.sigmoid(self.sigmoid_weights)
        W = tanh_weights * sigmoid_weights
        a = F.linear(input, W)
        return a


class NALU(nn.Module):
    def __init__(self, in_features: int, out_features: int, epsilon: float = 1e-12):
        super().__init__()
        self.nac = NAC(in_features, out_features)
        self.gate_weights = Parameter(torch.Tensor(out_features, in_features))
        self.epsilon = epsilon

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.gate_weights)

    def forward(self, input):
        g = torch.sigmoid(F.linear(input, self.gate_weights))
        a = self.nac(input)
        m = torch.exp(self.nac(torch.abs(input) + self.epsilon))

        nalu = g * a + (1 - g) * m
        return nalu


class PretrainedModel(nn.Module):
    def __init__(self, extract_intermediate_values: bool = False, n_dropout_runs: int = 0):
        super().__init__()
        self.extract_intermediate_values = extract_intermediate_values
        self.n_dropout_runs = n_dropout_runs
        self.pooled_features = None

        pretrained_model = densenet121(pretrained=True, drop_rate=0.0).train()
        self.extractor = pretrained_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(pretrained_model.classifier.in_features, 512),
            nn.ELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(pretrained_model.classifier.in_features, 512),
            nn.ELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            NAC(512, 1)
        )

        self.split_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(pretrained_model.classifier.in_features, 512),
            nn.ELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        if self.n_dropout_runs > 0:
            self.switch_dropouts_to_train()
            dropout_outputs = []
            for i in range(self.n_dropout_runs):
                outputs = self.single_forward(x,
                                              retain_pooled_features=True,
                                              use_retained_pooled_features=i > 0)
                dropout_outputs.append(outputs)
            self.switch_dropouts_to_previous_state()
            self.reset_pooled_features()

            averaged_outputs = [
                torch.stack(tuple([
                    dropout_outputs[j][i]
                    for j in range(self.n_dropout_runs)
                ]), dim=0).transpose(0, 1)
                for i in range(len(dropout_outputs[0]))
            ]

            return tuple(averaged_outputs)
        return self.single_forward(x,
                                   retain_pooled_features=False,
                                   use_retained_pooled_features=False)

    def switch_dropouts_to_train(self):
        for layer in self.modules():
            if isinstance(layer, nn.Dropout):
                layer.train()

    def switch_dropouts_to_previous_state(self):
        for layer in self.modules():
            if isinstance(layer, nn.Dropout):
                layer.train(self.training)

    def single_forward(self, x,
                       retain_pooled_features: bool = False,
                       use_retained_pooled_features: bool = False):
        is_test = len(x.size()) == 5
        if is_test:
            b, ncrops, c, h, w = x.size()

            if use_retained_pooled_features and self.pooled_features is not None:
                pooled_features = self.pooled_features
            else:
                features = self.extractor(x.view(-1, c, h, w))
                pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).view(
                    features.size(0), -1)

            class_out = self.classifier(pooled_features) \
                .view(b, ncrops, -1) \
                .mean(dim=1)
            regression_out = self.regressor(pooled_features) \
                .view(b, ncrops, -1) \
                .mean(dim=1)
            split_class_out = self.split_classifier(pooled_features) \
                .view(b, ncrops, -1) \
                .mean(dim=1)

        else:
            if use_retained_pooled_features and self.pooled_features is not None:
                pooled_features = self.pooled_features
            else:
                features = self.extractor(x)
                pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).view(
                    features.size(0), -1)

            class_out = self.classifier(pooled_features)
            regression_out = self.regressor(pooled_features)
            split_class_out = self.split_classifier(pooled_features)

        if retain_pooled_features:
            self.pooled_features = pooled_features
        else:
            self.pooled_features = None

        if self.extract_intermediate_values:
            if is_test:
                pooled_features = pooled_features.view(b, ncrops, -1).mean(dim=1)
            return class_out, regression_out, split_class_out, pooled_features

        return class_out, regression_out, split_class_out

    def reset_pooled_features(self):
        self.pooled_features = None
