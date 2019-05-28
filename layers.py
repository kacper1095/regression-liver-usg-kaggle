import numpy as np
import torch
import torch.nn as nn


class BatchRemaximalization(nn.Module):
    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "running_mean", "running_max",
                     "num_batches_tracked"]

    def __init__(self, num_features, eps=1e-5, momentum=0.1, track_running_stats=True):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(1, num_features, 1, 1))
            self.register_buffer("running_max", torch.ones(1, num_features, 1, 1))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_max", None)
            self.register_buffer("num_batches_tracked", None)

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_max.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()

    def extra_repr(self):
        return "eps={eps}, momentum={momentum}, track_running_stats={track_running_stats}".format(**self.__dict__)

    def _check_input_dim(self, input):
        pass

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1

                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        self.running_max.data = input.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0].mean(dim=0,
                                                                                                   keepdim=True) * exponential_average_factor + self.running_max * (
                                        1 - exponential_average_factor)
        self.running_mean.data = input.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) / (
                np.prod(input.shape[2:]) * input.shape[0]) * exponential_average_factor + self.running_mean * (
                                         1 - exponential_average_factor)

        normalized = (input - self.running_mean) / (self.running_max + self.eps)
        return normalized

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class FullyLearnableSigmoid(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.horizontal_squeeze = torch.nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.vertical_squeeze = torch.nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.horizontal_shift = torch.nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.vertical_shift = torch.nn.Parameter(torch.Tensor(1, num_features, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.horizontal_squeeze)
        torch.nn.init.ones_(self.vertical_squeeze)

        torch.nn.init.zeros_(self.horizontal_shift)
        torch.nn.init.zeros_(self.vertical_shift)

    def extra_repr(self):
        return "num_features={num_features}".format(**self.__dict__)

    def forward(self, input):
        return 1 / (1 + torch.exp(
            -self.horizontal_squeeze * (input - self.horizontal_shift))) * self.vertical_squeeze + self.vertical_shift

