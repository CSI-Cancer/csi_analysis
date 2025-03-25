import torch
import torch.nn as nn
import torch.nn.functional as F

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
            in_channels=4,
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
            growth_rate + num_layers_per_block * growth_rate, num_classes
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
        x = self.fc(x)
        return x