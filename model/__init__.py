from config import config
import copy
from model.base_model import BaseModel
import torch

__all__ = ['build_model']


def build_model(config):
    config = copy.deepcopy(config)
    module_class = BaseModel(config)
    return module_class


if __name__ == '__main__':
    model_config = config.Architecture
    model = build_model(model_config)
    x = torch.zeros(1, 3, 960, 960)
    print(model(x)['maps'].shape)
