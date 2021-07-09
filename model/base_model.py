from torch import nn
from model.backbone import build_backbone
from model.neck import build_neck
from model.head import build_head


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()

        # build backbone
        self.backbone = build_backbone(config["backbone"])
        in_channels = self.backbone.out_channels

        # build neck
        config['neck']['in_channels'] = in_channels
        self.neck = build_neck(config['neck'])
        in_channels = self.neck.out_channels

        # build head
        config["head"]['in_channels'] = in_channels
        self.head = build_head(config["head"])

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x
