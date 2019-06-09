import torch
import torch.nn as nn
import torch.nn.functional as F

class MnihCNN(nn.Module):
    def __init__(self):
        super(MnihCNN).__init__()
        self.conv1 = nn.Conv2d(3, 64, 16, stride=4, padding=0)
        self.conv2 = nn.Conv2d(64, 112, 4, stride=1, padding=0)
        self.conv3 = nn.Conv2d(112, 80, 3, stride=1, padding=0)
        self.fc4 = nn.Linear(3920, 4096)
        self.fc5 = nn.Linear(4096, 256)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pool2d(h, 2, 1)
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.fc4(h))
        h = self.fc5(h)
        h = F.reshape(h, (x.size()[0], 1, 16, 16))

