import copy
from .metric import DetMetric
__all__ = ['build_metric']


def build_metric(config):
    support_dict = ['DetMetric']

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'metric only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
