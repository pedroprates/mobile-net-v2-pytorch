from bottleneck import Bottleneck
from torch import nn


class MobileNetV2(nn.Module):
    def __init__(self, t, p, output_channels=1000, alpha=1, rho=1):
        super(MobileNetV2, self).__init__()

        self.initial_conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.initial_bn = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer('bottleneck_1', 1, 32, 16, 1, 1)
        self.layer2 = self._make_layer('bottleneck_2', t, 16, 24, 2, 2)
        self.layer3 = self._make_layer('bottleneck_3', t, 24, 32, 3, 2)
        self.layer4 = self._make_layer('bottleneck_4', t, 32, 64, 4, 2)
        self.layer5 = self._make_layer('bottleneck_5', t, 64, 96, 3, 1)
        self.layer6 = self._make_layer('bottleneck_6', t, 96, 160, 3, 2)
        self.layer7 = self._make_layer('bottleneck_7', t, 160, 320, 1, 1)

        self.pointwise1 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(1280)

        self.avg_pool = nn.AvgPool2d((7, 7))
        self.final_pointwise = nn.Conv2d(1280, output_channels, kernel_size=1)

        self.relu = nn.ReLU6()

        # Initialize conv and batchnorm layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu6')
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.pointwise1(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.avg_pool(x)

        x = self.final_pointwise(x)
        x = self.relu(x)

        return x

    @staticmethod
    def _make_layer(init_name, t, in_channels, out_channels, n, stride):
        layer = nn.Sequential()

        for i in range(n):
            s = stride if i == 0 else 1
            in_c = in_channels if i == 0 else out_channels

            layer.add_module(init_name + f'_{i+1}', Bottleneck(in_c, out_channels, t, s))

        return layer
