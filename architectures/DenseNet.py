import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet121(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNet121, self).__init__()
        self.net = models.densenet121(pretrained=True)
        channel_in = self.net.classifier.in_features
        classifier = nn.Sequential(
            nn.Linear(channel_in , num_classes)
        )
        self.net.classifier = classifier

    def forward(self, x):
        x = self.net(x)
        return x

def create_model():
    return DenseNet121()