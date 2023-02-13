import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    """
    Convolutional Neural Network.
    Applys specified convolution to the input image tensors and
    returns predictions
    """
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(6, 6), padding=(1, 1))
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = nn.Conv2d(
            128, 64, kernel_size=(6, 6), padding=(1, 1), stride=2)
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_3 = nn.Conv2d(
            64, 32, kernel_size=(6, 6), padding=(1, 1), stride=2)
        self.pool_3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.drop_1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2048, 120)
        self.fc2 = nn.Linear(120, 60)
        self.drop_2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(60, 1)

    def forward(self, images):
        x = F.relu(self.conv_1(images))
        x = self.pool_1(x)
        x = F.relu(self.conv_2(x))
        x = self.pool_2(x)
        x = F.relu(self.conv_3(x))
        x = self.pool_3(x)
        x = torch.flatten(x, 1)
        x = self.drop_1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop_2(x)
        x = self.fc3(x)
        return x
