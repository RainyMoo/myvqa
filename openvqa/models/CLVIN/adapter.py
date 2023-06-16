# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import torch.nn as nn
import torch
from openvqa.core.base_dataset import BaseAdapter
from openvqa.utils.make_mask import make_mask
from openvqa.models.CLVIN.Position import PositionEncoder

class Adapter(BaseAdapter):
    def __init__(self, __C):
        super(Adapter, self).__init__(__C)
        self.__C = __C

    def bbox_proc(self, bbox):
        area = (bbox[:, :, 2] - bbox[:, :, 0]) * (bbox[:, :, 3] - bbox[:, :, 1])
        return torch.cat((bbox, area.unsqueeze(2)), -1)

    def vqa_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][1]
        
        if __C.USE_BBOX_FEAT:
            self.posi = PositionEncoder(embed_dim=imgfeat_linear_size, posi_dim=6)
        self.frcn_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)

    def clevr_init(self, __C):
        self.grid_linear = nn.Linear(__C.FEAT_SIZE['clevr']['GRID_FEAT_SIZE'][1], __C.HIDDEN_SIZE)


    def vqa_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']

        img_feat_mask = make_mask(frcn_feat)
        
        # ori process with bbox_feat

        if self.__C.USE_BBOX_FEAT:
            w_feat = feat_dict['W_FEAT']
            h_feat = feat_dict['H_FEAT']
            w_feat = w_feat.unsqueeze(1)
            h_feat = h_feat.unsqueeze(1)
            wh_feat = torch.cat((w_feat, h_feat), dim=-1)
            pos_feat = self.posi(bbox_feat, wh_feat)
            frcn_feat = frcn_feat * pos_feat
            
        img_feat = self.frcn_linear(frcn_feat)

        return img_feat, img_feat_mask


    def clevr_forward(self, feat_dict):
        grid_feat = feat_dict['GRID_FEAT']

        img_feat_mask = make_mask(grid_feat)
        img_feat = self.grid_linear(grid_feat)

        return img_feat, img_feat_mask



