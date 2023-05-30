import torch
import torch.nn as nn
import torchvision.models as models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class Net(nn.Module):
    def __init__(self, neurons=256, dropout=0.5):
        super(Net, self).__init__()
        self.net = models.vgg16_bn(pretrained=True) # class Net has an attribute net
        for param in self.net.parameters():
            param.requires_grad = False
        in_features = 25088
        self.net.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, neurons),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(neurons, 1)
        )

    def forward(self,x):
        return self.net(x)