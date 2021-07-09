import errno
import os
import torch


def _mkdir_if_not_exist(path, logger):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning('be happy if some process has already created {}'.format(path))
            else:
                raise OSError('Failed to mkdir {}'.format(path))


def init_model(config, model, logger, optimizer=None, lr_scheduler=None):
    checkpoints = config.get('checkpoints')
    pretrained_model = config.get('pretrained_model')
    best_model_dict = {}
    if checkpoints:
        checkpoint = torch.load(checkpoints)
        model_dict = checkpoint['model']
        model.load_state_dict(model_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'global_step' in checkpoint:
            lr_scheduler.last_epoch = checkpoint['global_step']
        best_model_dict = model_dict.get('best_model_dict', {})
        best_model_dict['start_epoch'] = checkpoint['epoch'] + 1
        logger.info("resume from {}".format(checkpoints))
    elif pretrained_model:
        logger.info("load pretrained model from {}".format(pretrained_model))
    else:
        logger.info('train from scratch')
    return best_model_dict


def save_model(model, optimizer, model_path, logger, prefix='best_accuracy', epoch=0, global_step=-1, **kwargs):
    _mkdir_if_not_exist(model_path, logger)
    model_name = os.path.join(model_path, prefix + '.pth')
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch,
             'global_step': global_step}
    torch.save(state, model_name)
