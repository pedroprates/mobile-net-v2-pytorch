from torch import nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride=1):
        super(Bottleneck, self).__init__()
        middle_channels = expansion * in_channels
        self.residual = stride == 1 and in_channels == out_channels

        self.pointwise_in = nn.Conv2d(in_channels,
                                      middle_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=1)

        self.bn1 = nn.BatchNorm2d(middle_channels)

        self.depthwise = nn.Conv2d(middle_channels,
                                   middle_channels,
                                   kernel_size=3,
                                   stride=stride,
                                   padding=1,
                                   groups=middle_channels)

        self.bn2 = nn.BatchNorm2d(middle_channels)

        self.pointwise_out = nn.Conv2d(middle_channels,
                                       out_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=1)

        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU6()

    def forward(self, x):
        initial = x

        out_p1 = self.pointwise_in(x)
        out_p1 = self.bn1(out_p1)
        out_p1 = self.relu(out_p1)

        out_p2 = self.depthwise(out_p1)
        out_p2 = self.bn2(out_p2)
        out_p2 = self.relu(out_p2)

        out = self.pointwise_out(out_p2)
        out = self.bn3(out)
        out = self.relu(out)

        if self.residual:
            out += initial

        return out
