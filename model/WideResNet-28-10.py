import torch
import torch.nn as nn
import torch.nn.functional as F

# WideResNet implementation
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.equalInOut = (in_planes == out_planes)
        if not self.equalInOut:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                      stride=stride, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.relu(self.bn1(x))
        if not self.equalInOut:
            x = out
        out = self.conv1(out)
        out = self.dropout(self.relu(self.bn2(out)))
        out = self.conv2(out)
        return out + (x if self.shortcut is None else self.shortcut(x))


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout_rate):
        super().__init__()
        self.layer = self._make_layer(nb_layers, in_planes, out_planes, block, stride, dropout_rate)

    def _make_layer(self, nb_layers, in_planes, out_planes, block, stride, dropout_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes,
                                out_planes,
                                i == 0 and stride or 1,
                                dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.0, num_classes=100):
        super().__init__()
        n = (depth - 4) // 6
        k = widen_factor

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)

        self.block1 = NetworkBlock(n, 16, 16*k, BasicBlock, 1, dropout_rate)
        self.block2 = NetworkBlock(n, 16*k, 32*k, BasicBlock, 2, dropout_rate)
        self.block3 = NetworkBlock(n, 32*k, 64*k, BasicBlock, 2, dropout_rate)

        self.bn = nn.BatchNorm2d(64*k)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(64*k, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(out.size(0), -1)
        return self.fc(out)
