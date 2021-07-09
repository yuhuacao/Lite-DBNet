import copy
from torch.utils.data import DataLoader
from data.dataset import ICDAR2015DataSet


def build_dataloader(config, mode, logger):
    support_dict = ['ICDAR2015DataSet']
    module_name = config.Train['dataset']['name']
    assert module_name in support_dict, Exception('DataSet only support {}'.format(support_dict))
    assert mode in ['train', 'test'], "Mode should be Train or Test."
    dataset = eval(module_name)(config, mode, logger)
    if mode == 'train':
        train_config = copy.deepcopy(config.Train)
        loader_config = train_config['dataloader']
    else:
        test_config = copy.deepcopy(config.Test)
        loader_config = test_config['dataloader']

    data_loader = DataLoader(dataset,
                             batch_size=loader_config['batch_size'],
                             shuffle=loader_config['shuffle'],
                             num_workers=loader_config['num_workers'],
                             drop_last=loader_config['drop_last'])

    return data_loader
