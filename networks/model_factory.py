import torch.nn as nn

from networks.resnet import resnet18
from networks.shufflenet import shufflenet_v2_x1_0
from networks.cifar_net import Net
from networks.mlp import MLP


class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(target_model, num_classes, img_size, pretrained=False):

        if target_model == 'mlp': 
            return MLP(feature_size=img_size, hidden_dim=40, num_class=num_classes)

        elif target_model == 'resnet':
            if pretrained:
                model = resnet18(pretrained=True)
                model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
            else:
                model = resnet18(pretrained=False, num_classes=num_classes)
            return model

        elif target_model == 'cifar_net':
            return Net(num_classes=num_classes)

        elif target_model == 'shufflenet':
            if pretrained:
                model = shufflenet_v2_x1_0(pretrained=True)
                model.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
            else:
                model = shufflenet_v2_x1_0(pretrained=False, num_classes=num_classes)
            return model

        else:
            raise NotImplementedError

