from optimizer import lr_scheduler
import copy


def build_lr_scheduler(optimizer, lr_config, epochs, step_each_epoch):
    lr_config = copy.deepcopy(lr_config)
    lr_config.update({'epochs': epochs, 'step_each_epoch': step_each_epoch})
    if 'name' in lr_config:
        lr_name = lr_config.pop('name')
        lr = getattr(lr_scheduler, lr_name)(optimizer, **lr_config)
    else:
        lr = lr_config['learning_rate']
    return lr
