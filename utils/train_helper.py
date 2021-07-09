import torch
import os
from utils.stats import TrainingStats
import time
from tqdm import tqdm
from utils.save_load import save_model
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()


def train(config,
          train_dataloader,
          valid_dataloader,
          device,
          model,
          loss_class,
          optimizer,
          lr_scheduler,
          post_process_class,
          eval_class,
          pre_best_model_dict,
          logger,
          ):  # vdl_writer=None

    log_smooth_window = config['log_smooth_window']
    epoch_num = config['epoch_num']
    print_batch_step = config['print_batch_step']
    eval_batch_epoch = config['eval_batch_epoch']

    global_step = 0
    start_eval_step = 0
    if type(eval_batch_epoch) == list and len(eval_batch_epoch) >= 2:
        start_eval_epoch = eval_batch_epoch[0]
        eval_batch_epoch = eval_batch_epoch[1]
        if len(valid_dataloader) == 0:
            logger.info('No Images in eval dataset, evaluation during training will be disabled.')
            start_eval_step = 1e111
        logger.info("During the training process, after the {}th epoch, an evaluation is run every {} epoch.".
                    format(start_eval_step, eval_batch_epoch))

    save_epoch_step = config['save_epoch_step']
    save_model_dir = config['save_model_dir']
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    main_indicator = eval_class.main_indicator
    best_model_dict = {main_indicator: 0}
    best_model_dict.update(pre_best_model_dict)

    train_stats = TrainingStats(log_smooth_window, ['lr'])
    # model_average = False
    model.train()

    if 'start_epoch' in best_model_dict:
        start_epoch = best_model_dict['start_epoch']
    else:
        start_epoch = 1

    for epoch in range(start_epoch, epoch_num + 1):
        train_batch_cost = 0.0
        train_reader_cost = 0.0
        batch_sum = 0
        batch_start = time.time()
        for idx, batch in enumerate(train_dataloader):
            train_reader_cost += time.time() - batch_start
            if idx >= len(train_dataloader):
                break
            optimizer.zero_grad()
            batch = [bat.to(device) for bat in batch]
            images = batch[0]
            with autocast():
                preds = model(images)
                loss = loss_class(preds, batch)
            avg_loss = loss['loss']
            scaler.scale(avg_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            train_batch_cost += time.time() - batch_start
            batch_sum += len(images)

            # logger
            stats = {k: v.cpu().detach().numpy().mean() for k, v in loss.items()}
            stats['lr'] = optimizer.param_groups[0]['lr']
            train_stats.update(stats)

            if global_step > 0 and global_step % print_batch_step == 0:
                logs = train_stats.log()
                strs = 'epoch: [{}/{}], iter: {}, {}, reader_cost: {:.5f} s, batch_cost: {:.5f} s, samples: {}, ' \
                       'ips: {:.5f}'.format(epoch, epoch_num, global_step, logs, train_reader_cost / print_batch_step,
                                            train_batch_cost / print_batch_step, batch_sum,
                                            batch_sum / train_batch_cost)
                logger.info(strs)
                train_batch_cost = 0.0
                train_reader_cost = 0.0
                batch_sum = 0

            global_step += 1
            batch_start = time.time()

        # eval
        if epoch > start_eval_epoch and (global_step - start_eval_step) % eval_batch_epoch == 0:
            cur_metric = eval(model, valid_dataloader, post_process_class, eval_class, device)
            cur_metric_str = 'cur metric, {}'.format(', '.join(
                ['{}: {:.4f}'.format(k, v) for k, v in cur_metric.items()]))
            logger.info(cur_metric_str)

            if cur_metric[main_indicator] >= best_model_dict[main_indicator]:
                best_model_dict.update(cur_metric)
                best_model_dict['best_epoch'] = epoch
                save_model(
                    model,
                    optimizer,
                    save_model_dir,
                    logger,
                    is_best=True,
                    prefix='best_accuracy',
                    last_step=global_step,
                    epoch=epoch)
            best_str = 'best metric, {}'.format(', '.join(['{}: {:.4f}'.format(k, v) for k, v in best_model_dict.items()]))
            logger.info(best_str)

    best_str = 'best metric, {}'.format(', '.join(['{}: {}'.format(k, v) for k, v in best_model_dict.items()]))
    logger.info(best_str)
    return


def eval(model, valid_dataloader, post_process_class, eval_class, device):
    model.eval()
    with torch.no_grad():
        total_frame = 0.0
        total_time = 0.0
        for idx, batch in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader), desc='eval model:'):
            if idx >= len(valid_dataloader):
                break
            batch = [bat.to(device) for bat in batch]
            images = batch[0]
            start = time.time()

            preds = model(images)

            batch = [item.cpu().numpy() for item in batch]
            # Obtain usable results from post-processing methods
            post_result = post_process_class(preds, batch[1])
            total_time += time.time() - start
            # Evaluate the results of the current batch
            eval_class(post_result, batch)
            total_frame += len(images)
        # Get final metric，eg. acc or hmean
        metric = eval_class.get_metric()

    model.train()
    metric['fps'] = total_frame / total_time
    return metric
