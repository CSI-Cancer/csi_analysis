import torch
import torch.nn as nn
import torch.nn.functional as F

# Classes are returned as an incrementing values from 0
CLASSES = [
    "D",
    "CK",
    "CD",
    "V",
    "CK|CD|V",
    "CK|CD",
    "D|CK|CD|V",
    "CK|V",
    "D|CK|CD",
    "D|CK|V",
    "D|V",
    "D|CD|V",
    "D|CD",
    "D|CK",
    "CD|V",
]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.5):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dropout_rate=0.5):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(
            block, 64, num_blocks[0], stride=1, dropout_rate=dropout_rate
        )
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], stride=2, dropout_rate=dropout_rate
        )
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], stride=2, dropout_rate=dropout_rate
        )
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], stride=2, dropout_rate=dropout_rate
        )
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, dropout_rate))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, dropout_rate=0.5):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout_rate = dropout_rate
        for i in range(num_layers):
            self.layers.append(
                self._make_layer(in_channels + i * growth_rate, growth_rate)
            )

    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.Dropout(self.dropout_rate),
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], 1)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.layer(x)


class DenseNet(nn.Module):
    def __init__(
        self, num_classes=15, growth_rate=32, num_layers_per_block=4, dropout_rate=0.5
    ):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.num_layers_per_block = num_layers_per_block
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=2 * growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(2 * growth_rate)

        self.block1 = DenseBlock(
            2 * growth_rate, growth_rate, num_layers_per_block, dropout_rate
        )
        self.trans1 = TransitionLayer(
            2 * growth_rate + num_layers_per_block * growth_rate, growth_rate
        )

        self.block2 = DenseBlock(
            growth_rate, growth_rate, num_layers_per_block, dropout_rate
        )
        self.trans2 = TransitionLayer(
            growth_rate + num_layers_per_block * growth_rate, growth_rate
        )

        self.block3 = DenseBlock(
            growth_rate, growth_rate, num_layers_per_block, dropout_rate
        )
        self.trans3 = TransitionLayer(
            growth_rate + num_layers_per_block * growth_rate, growth_rate
        )

        self.block4 = DenseBlock(
            growth_rate, growth_rate, num_layers_per_block, dropout_rate
        )

        self.bn2 = nn.BatchNorm2d(growth_rate + num_layers_per_block * growth_rate)
        self.fc = nn.Linear(
            1+growth_rate + num_layers_per_block * growth_rate, num_classes
        )

    def forward(self, x, prefix):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = self.trans3(x)
        x = self.block4(x)
        x = F.relu(self.bn2(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, prefix.unsqueeze(1)], 1)
        x = self.fc(x)
        return x
    
class GenericCNN(nn.Module):
    def __init__(self, dropout=0.2, num_classes=15):
        super(GenericCNN, self).__init__()
        # Convolutional layers adjustments
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # Adjusted output channels
        self.bn4 = nn.BatchNorm2d(512)  # Adjusted for 512 output channels
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)  # Adjusted output channels
        self.bn5 = nn.BatchNorm2d(512)  # Adjusted for 512 output channels
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(513, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)



    def forward(self, x, prefix):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = x.view(-1, 512)  # Flatten the tensor
        x = torch.cat([x, prefix.unsqueeze(1)], 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


def ResNet4(dropout=0.5, num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, dropout=dropout)


def DenseNet121(dropout, num_classes):
    return DenseNet(num_classes=num_classes,
                    growth_rate=32,
                    num_layers_per_block=6,
                    dropout_rate=dropout)

def get_model(dropout=0.5, num_classes=15, model_name="generic"):
    if model_name == "resnet":
        return ResNet4(dropout=dropout, num_classes=num_classes)
    elif model_name == "densenet":
        return DenseNet121(dropout=dropout, num_classes=num_classes)
    elif model_name == "generic":
        return GenericCNN(dropout=dropout, num_classes=num_classes)
    else:
        raise ValueError("Unknown model name")
