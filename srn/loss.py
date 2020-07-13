""" Custom loss function """

import torch

class BalancedBCELoss(torch.nn.Module):
    def __init__(self, weights):
        super(BalancedBCELoss, self).__init__()
        self.weights = weights

    def forward(self, x, target):
        x = torch.sigmoid(x)
        _output = torch.clamp(x, min=1e-6, max=(1 - 1e-6))
        loss = self.weights[0] * (target * torch.log(_output))\
            +  self.weights[1] * ((1 - target) * torch.log(1 - _output))
        return torch.neg(torch.mean(loss))
