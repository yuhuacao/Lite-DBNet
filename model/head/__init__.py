from .db_head import DBHead


def build_head(config):
    module_name = config.pop('name')
    module_class = eval(module_name)(**config)
    return module_class
