import copy
from postprocess.db_postprocess import DBPostProcess

__all__ = ['build_post_process']


def build_post_process(config, global_config=None):
    support_dict = ['DBPostProcess']

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        'post process only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
