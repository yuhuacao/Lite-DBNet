import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
from neck.db_fpn import DBFPN


def build_neck(config):
    module_name = config.pop('name')
    module_class = eval(module_name)(**config)
    return module_class
