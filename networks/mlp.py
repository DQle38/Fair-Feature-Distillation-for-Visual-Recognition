import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class MLP(nn.Module):
    def __init__(self, feature_size, hidden_dim, num_class=None, num_layer=2, adv=False, adv_lambda=1.):
        super(MLP, self).__init__()
        try:
            in_features = self.compute_input_size(feature_size)  # if list
        except:
            in_features = feature_size  # if int

        self.adv = adv
        if self.adv:
            self.adv_lambda = adv_lambda

        self.num_layer = num_layer

        fc = []
        in_dim = in_features
        for i in range(num_layer-1):
            fc.append(nn.Linear(in_dim, hidden_dim))
            fc.append(nn.ReLU())
            in_dim = hidden_dim

        fc.append(in_dim, num_class)
        self.fc = nn.Sequential(*fc)

    def forward(self, feature):
        feature = torch.flatten(feature, 1)
        if self.adv:
            feature = ReverseLayerF.apply(feature, self.adv_lambda)

        out = self.fc(feature)

        return out

    def compute_input_size(self, feature_size):
        in_features = 1
        for size in feature_size:
            in_features = in_features * size

        return in_features
    

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
