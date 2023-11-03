# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from importlib import import_module


class ModelLoader:
    def __init__(self, __C):

        self.model_use = __C.MODEL_USE
        model_moudle_path = 'openvgd.models.' + self.model_use + '.full_vgd'
        self.model_moudle = import_module(model_moudle_path)

    def Net_Full(self, __arg1, __arg2):
        return self.model_moudle.Net_Full(__arg1, __arg2)

'''
class CfgLoader:
    def __init__(self, model_use):

        cfg_moudle_path = 'openvgd.models.' + model_use + '.model_cfgs'
        self.cfg_moudle = import_module(cfg_moudle_path)

    def load(self):
        return self.cfg_moudle.Cfgs()
'''