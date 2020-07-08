import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(dims, 8)
        self.out = nn.Linear(8, 4)
        self.act = nn.ReLU()

    def forward(self, x):

        x = F.relu(self.hidden(x))
        x = self.act(self.out(x))

        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)

        self.dense = nn.Linear(2304, 16)
        self.batchnorm4 = nn.BatchNorm1d(16)
        self.drop = nn.Dropout(0.5)

        self.out = nn.Linear(16, 4)
        self.act = nn.ReLU()

    def forward(self, x):

        x = F.relu(self.conv1(x))  # [batchsize, 16, 62, 62]
        x = self.batchnorm1(x)  # [batchsize, 16, 62, 62]
        x = self.pool1(x)  # [batchsize, 16, 31, 31]

        x = F.relu(self.conv2(x))  # [batchsize, 32, 29, 29]
        x = self.batchnorm2(x)  # [batchsize, 32, 29, 29]
        x = self.pool2(x)  # [batchsize, 32, 14, 14]

        x = F.relu(self.conv3(x))  # [batchsize, 64, 12, 12]
        x = self.batchnorm3(x)  # [batchsize, 64, 12, 12]
        x = self.pool3(x)  # [batchsize, 64, 6, 6]

        x = x.view(x.size(0), -1)  # [batchsize, 2304]
        x = F.relu(self.dense(x))  # [batchsize, 16]
        x = self.batchnorm4(x)  # [batchsize, 16]
        x = self.drop(x)  # [batchsize, 16]

        x = self.act(self.out(x))  # [batchsize, 4]

        return x


class Frankenstein(nn.Module):
    """docstring for Frankenstein"""
    def __init__(self, dims):
        super(Frankenstein, self).__init__()
        self.cnn = CNN()
        self.mlp = MLP(dims)

        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, image, data):
        # [1, 4]
        x1 = self.cnn(image)
        x2 = self.mlp(data)
        x = torch.cat((x1, x2), dim=1)

        x = F.relu(self.fc1(x))  # [1, 8]
        x = self.sigmoid(self.fc2(x))

        return x


if __name__ == '__main__':

    batch_size, C, H, W = 1, 3, 64, 64
    net = Frankenstein(5)
    x = torch.randn(batch_size, C, H, W)
    y = torch.randn(batch_size, 10)
    net.eval()
    output = net(x, y)
    print(net)
