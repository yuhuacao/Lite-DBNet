import copy
from loss.db_loss import DBLoss


def build_loss(config):
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    module_class = eval(module_name)(**config)
    return module_class
