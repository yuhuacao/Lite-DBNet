import torch
from torch import nn
from torch.hub import load_state_dict_from_url

__all__ = ['MobileNetV2', 'mobilenet_v2']
model_urls = {'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'}


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, block=None, norm_layer=None):
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.layer1 = ConvBNReLU(3, input_channel, stride=2)

        self.stages = []
        self.out_channels = []
        block_list = []
        idx = 0

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                if stride == 2 and idx > 1:
                    self.out_channels.append(input_channel)
                    self.stages.append(nn.Sequential(*block_list))
                    block_list = []
                block_list.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
            idx += 1
        block_list.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.stages.append(nn.Sequential(*block_list))
        self.out_channels.append(last_channel)
        self.layer2 = self.stages[0]
        self.layer3 = self.stages[1]
        self.layer4 = self.stages[2]
        self.layer5 = self.stages[3]
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layer1(x)
        out_list = []
        for stage in self.stages:
            x = stage(x)
            out_list.append(x)
        return out_list


def mobilenet_v2(pretrained=True, progress=True, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


if __name__ == "__main__":
    x = torch.zeros(1, 3, 960, 960).to('cuda')
    # model = MobileNetV2().to('cuda')
    model = mobilenet_v2(pretrained=True).cuda()
    print(model.out_channels)
    for c in model(x):
        print(c.shape)
