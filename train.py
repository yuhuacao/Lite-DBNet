import torch
from data import build_dataloader
from model import build_model
from optimizer import build_lr_scheduler
from metric import build_metric
from config import config
from utils.logging import get_logger
from postprocess import build_post_process
from loss import build_loss
from utils import train_helper
from utils.save_load import init_model
import warnings
warnings.filterwarnings('ignore')


def main(config, logger):
    # config信息
    global_config = config.Global
    model_config = config.Architecture
    loss_config = config.Loss
    post_config = config.DBPostProcess
    optim_config = config.Optimizer
    metric_config = config.Metric
    device = "cuda" if global_config['use_gpu'] else "cpu"  # 检查设备

    # build dataloader
    train_dataloader = build_dataloader(config, 'train', logger)
    test_dataloader = build_dataloader(config, 'test', logger)
    logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
    if test_dataloader is not None:
        logger.info('valid dataloader has {} iters'.format(len(test_dataloader)))

    # build post process
    post_process_class = build_post_process(post_config, global_config)

    # build model
    model = build_model(model_config)
    model = model.to(device)

    # build loss
    criterion = build_loss(loss_config).to(device)

    # build optim
    optim = torch.optim.Adam(params=model.parameters(),
                             lr=optim_config['learning_rate'],
                             betas=optim_config['beta'],
                             weight_decay=optim_config['weight_decay'],
                             amsgrad=True)
    lr_scheduler = build_lr_scheduler(optimizer=optim,
                                      lr_config=optim_config['lr'],
                                      epochs=global_config['epoch_num'],
                                      step_each_epoch=len(train_dataloader))

    # build metric
    eval_class = build_metric(metric_config)

    # load pretrain model
    pre_best_model_dict = init_model(global_config, model, logger, optim)

    # start train
    train_helper.train(global_config, train_dataloader, test_dataloader, device, model, criterion, optim, lr_scheduler,
                       post_process_class, eval_class, pre_best_model_dict, logger)


if __name__ == '__main__':
    # config, device, logger, vdl_writer = program.preprocess(is_train=True)
    global_config = config.Global
    config = config
    log_file = '{}train.log'.format(global_config['save_model_dir'])
    print(log_file)
    logger = get_logger(name='root', log_file=log_file)
    main(config, logger)
