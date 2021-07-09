from torch.optim.lr_scheduler import _LRScheduler
import math


class Cosine(_LRScheduler):
    """
    WarmUP + Cosine learning rate decay
    """

    def __init__(self,
                 optimizer,
                 step_each_epoch,
                 epochs,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        # self.learning_rate = learning_rate
        self.T_max = step_each_epoch * epochs
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)
        super(Cosine, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epoch:
            factor = self.last_epoch / self.warmup_epoch
        else:
            factor = math.cos((self.last_epoch - self.warmup_epoch) * math.pi / 2 / (self.T_max - self.warmup_epoch))
        return [base_lr * factor for base_lr in self.base_lrs]


if __name__ == '__main__':
    import torch
    from torchvision.models import resnet18
    from matplotlib import pyplot as plt

    max_iter = 200 * 10
    model = resnet18()
    optimizer = torch.optim.SGD(model.parameters(), 1e-3)
    sc = Cosine(optimizer, learning_rate=0.001, step_each_epoch=10, epochs=200, warmup_epoch=20)
    lr = []
    for i in range(max_iter):
        sc.step()
        print(i, sc.last_epoch, sc.get_lr()[0])
        lr.append(sc.get_lr()[0])

    plt.plot(list(range(max_iter)), lr)
    plt.show()
