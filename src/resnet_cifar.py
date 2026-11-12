"""CIFAR ResNet (He et al. 2016, Section 4.2): depth 6n+2, 16/32/64 filters, option-A shortcuts."""
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.pad = planes - in_planes if (stride != 1 or in_planes != planes) else 0

    def shortcut(self, x):
        if self.pad == 0:
            return x
        # Option A: subsample spatially and zero-pad the extra channels (parameter-free).
        x = x[:, :, ::2, ::2]
        return F.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CifarResNet(nn.Module):
    def __init__(self, depth=110, num_classes=100):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2"
        n = (depth - 2) // 6
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        layers, in_planes = [], 16
        for planes, stride in [(16, 1), (32, 2), (64, 2)]:
            for i in range(n):
                layers.append(BasicBlock(in_planes, planes, stride if i == 0 else 1))
                in_planes = planes
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)
