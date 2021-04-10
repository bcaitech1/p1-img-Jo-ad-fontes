import torch
import torch.nn as nn
import torch.nn.init as init

from config import get_args

from torchvision.models import resnext50_32x4d, resnet34, resnet50
from efficientnet_pytorch import EfficientNet


def initialize_weights(model):
    """
    Initialize all weights using xavier uniform.
    For more weight initialization methods, check https://pytorch.org/docs/stable/nn.init.html
    """

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    return model


def get_res_pre_trained(model_name):
    efficientnets = [
        "efficientnet-b0",
        "efficientnet-b1",
        "efficientnet-b2",
        "efficientnet-b3",
        "efficientnet-b4",
        "efficientnet-b5",
        "efficientnet-b6",
        "efficientnet-b7",
    ]

    if model_name == "resnext50":
        model = resnext50_32x4d(True)
        num_classes = 18

        fc_in_features = model.fc.in_features

        model.fc = nn.Linear(in_features=fc_in_features, out_features=num_classes)

    elif model_name == "resnet34":
        model = resnet34(True)

    elif model_name in efficientnets:

        model = EfficientNet.from_pretrained(model_name, num_classes=18)

    elif model_name == "resnet50_cutmix_v2":
        model = resnet50(True)
        model.load_state_dict(torch.load("/opt/ml/_model/ResNet50_CutMix_v2.pth"))

    elif model_name == "resnet50":
        model = resnet50(True)
        num_classes = 18
        model.fc = nn.Linear(in_features=2048, out_features=num_classes)

    else:
        model = nn.Module()
    return model


if __name__ == "__main__":
    pass
