import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
from config import config
import torch
from backbone.mobilenet_v2 import mobilenet_v2
from backbone.mobilenet_v3 import mobilenet_v3_large, mobilenet_v3_small
__all__ = ['build_backbone']


def build_backbone(config):
    support_dict = ['mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small']

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('backbone only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class


if __name__ == "__main__":
    backbone_config = config.Architecture['backbone']
    model = build_backbone(backbone_config)
    x = torch.zeros(1, 3, 960, 960)
    print(model(x))
