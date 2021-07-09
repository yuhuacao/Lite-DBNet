import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
from torch.utils.data import Dataset, DataLoader
from config import config
from data.imaug import transform, create_operators
from utils.logging import get_logger


class ICDAR2015DataSet(Dataset):
    def __init__(self, config, mode, logger):
        super(ICDAR2015DataSet, self).__init__()
        self.logger = logger
        assert mode in ['train', 'test'], "The mode must be 'train' or 'test'."
        if mode == 'train':
            dataset_config = config.Train['dataset']
        else:
            dataset_config = config.Test['dataset']
        label_file = dataset_config['label_file']
        global_config = config.Global

        self.data_dir = dataset_config['data_dir']
        self.data_lines = self.get_image_info_list(self, label_file)

        self.ops = create_operators(dataset_config['transforms'], global_config)

    @staticmethod
    def get_image_info_list(self, file):
        with open(file, "rb") as f:
            data_lines = f.readlines()
        return data_lines

    def __getitem__(self, idx):
        data_line = self.data_lines[idx]
        try:
            data_line = data_line.decode('utf-8')
            file_name, label = data_line.strip("\n").split('\t')
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            outs = transform(data, self.ops)
        except Exception as e:
            outs = None
            self.logger.error("When parsing line {}, error happened with msg: {}".format(data_line, e))
        else:
            return outs

    def __len__(self):
        return len(self.data_lines)


if __name__ == '__main__':
    global_config = config.Global
    log_file = '{}/train.log'.format(global_config['save_model_dir'])
    logger = get_logger(name='root', log_file=log_file)
    loader_config = config.Test['dataloader']
    train_dataset = ICDAR2015DataSet(config, 'test', logger)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=loader_config['batch_size'],
                                  shuffle=loader_config['shuffle'],
                                  num_workers=loader_config['num_workers'],
                                  drop_last=loader_config['drop_last'])
    data = train_dataset[0]
    print(data)
    for idx, batch in enumerate(train_dataloader):
        # for bat in batch:
        #     print(bat.shape)
        print(batch[0].to('cuda'))
        break


