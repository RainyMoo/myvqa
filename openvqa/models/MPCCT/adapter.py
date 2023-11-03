# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------
from openvqa.ops.fc import FC, MLP
import torch.nn as nn
import torch
from openvqa.core.base_dataset import BaseAdapter
from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.MPCCT.position import RegionAbsolutePosition, GridRelationalEmbedding, PositionEmbeddingSine, RegionRelationalEmbedding
import torch.nn.functional as F
import math

class Adapter(BaseAdapter):
    def __init__(self, __C):
        super(Adapter, self).__init__(__C)
        self.__C = __C

    def bbox_proc(self, boxes, imgs_wh):
        xmin = boxes[:, :, 0]
        ymin = boxes[:, :, 1] 
        xmax = boxes[:, :, 2] 
        ymax = boxes[:, :, 3]
        area = (boxes[:, :, 2] - boxes[:, :, 0]) * (boxes[:, :, 3] - boxes[:, :, 1])
        expand_wh = torch.cat([imgs_wh, imgs_wh], dim=1).unsqueeze(dim=1)    #(bs, 1, 4)
        ratio_area = area / (imgs_wh[:, 0] * imgs_wh[:, 1]).unsqueeze(-1) #(bs, num_r)
        ratio_area = ratio_area.unsqueeze(-1) #(bs, num_r, 1)
        boxes = torch.stack([xmin, ymin, xmax, ymax], dim=2)
        boxes = boxes / expand_wh   #(bs, num_r, 4)
        res = torch.cat([boxes, ratio_area], dim=-1)  #(bs, num_r, 5)
        return res

    def vqa_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][1]
        #if __C.USE_BBOX_FEAT:
            #self.bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
            #imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.frcn_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)    
        self.linear_abs = nn.Linear(6, __C.HIDDEN_SIZE)

    def vqa_grid_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['vqa_grid']['FRCN_FEAT_SIZE'][1]
        self.frcn_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)
        #if __C.USE_BBOX_FEAT:
            #self.bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
            #imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.Abs = PositionEmbeddingSine(256, normalize=True)
        
    def gqa_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['gqa']['FRCN_FEAT_SIZE'][1]
        if __C.USE_BBOX_FEAT:
            self.bbox_linear = nn.Linear(6, __C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.frcn_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)
        self.linear_abs = nn.Linear(6, __C.HIDDEN_SIZE)

        if __C.USE_AUX_FEAT:
            self.grid_linear = nn.Linear(__C.FEAT_SIZE['gqa']['GRID_FEAT_SIZE'][1], __C.HIDDEN_SIZE)

    def clevr_init(self, __C):
        self.grid_linear = nn.Linear(__C.FEAT_SIZE['clevr']['GRID_FEAT_SIZE'][1], __C.HIDDEN_SIZE)
        self.grid_abs = PositionEmbeddingSine(256, normalize=True)

    def vqa_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']
        w_feat = feat_dict['W_FEAT']
        h_feat = feat_dict['H_FEAT']
        w_feat = w_feat.unsqueeze(1)
        h_feat = h_feat.unsqueeze(1)
        wh_feat = torch.cat((w_feat, h_feat), dim=-1)
        img_feat_mask = make_mask(frcn_feat)

        #if self.__C.USE_BBOX_FEAT:
            #bbox_feat = self.bbox_proc(bbox_feat)
            #bbox_feat = self.bbox_linear(bbox_feat)
            #frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        img_feat = self.frcn_linear(frcn_feat)
        region_abs = RegionAbsolutePosition(bbox_feat, wh_feat) # (bs, num, 6)
        region_abs = self.linear_abs(region_abs)   
        region_rel = RegionRelationalEmbedding(boxes=bbox_feat,
                                             dim_g=64, 
                                             wave_len=1000, 
                                             trignometric_embedding=True) # (bs, num, num, 6)
        

        return img_feat, img_feat_mask, region_abs, region_rel

    def vqa_grid_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']

        img_feat_mask = make_mask(frcn_feat)
        
        img_feat = self.frcn_linear(frcn_feat)

        bs = img_feat.shape[0]
        grid_abs = self.Abs(img_feat.view(bs, 8, 8, -1))   # (bs, num, dim)
        grid_rel = GridRelationalEmbedding(grid_feat=img_feat,
                                           grid_size=8, 
                                           dim_g=64, 
                                           wave_len=1000, 
                                           trignometric_embedding=True)  #(bs * r * r *6)

        return img_feat, img_feat_mask, grid_abs, grid_rel 
    
    def gqa_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']
        grid_feat = feat_dict['GRID_FEAT']
        w_feat = feat_dict['W_FEAT']
        h_feat = feat_dict['H_FEAT']
        w_feat = w_feat.unsqueeze(1)
        h_feat = h_feat.unsqueeze(1)
        wh_feat = torch.cat((w_feat, h_feat), dim=-1)

        img_feat_mask = make_mask(frcn_feat)
        region_abs = RegionAbsolutePosition(bbox_feat, wh_feat) # (bs, num, 6)
        region_rel = RegionRelationalEmbedding(bbox_feat)

        if self.__C.USE_BBOX_FEAT:
            #bbox_feat = self.bbox_proc(bbox_feat, wh_feat)
            bbox_feat = self.bbox_linear(region_abs)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        img_feat = self.frcn_linear(frcn_feat)

        if self.__C.USE_AUX_FEAT:
            grid_feat_mask = make_mask(grid_feat)
            img_feat_mask = torch.cat((img_feat_mask, grid_feat_mask), dim=-1)
            grid_feat = self.grid_linear(grid_feat)
            img_feat = torch.cat((img_feat, grid_feat), dim=1)

        #region_abs = RegionAbsolutePosition(bbox_feat, wh_feat) # (bs, num, 6)
        region_abs = self.linear_abs(region_abs)   
        #region_rel = RegionRelationalEmbedding(bbox_feat) # (bs, num, num, 6)

        return img_feat, img_feat_mask, region_abs, region_rel


    def clevr_forward(self, feat_dict):
        grid_feat = feat_dict['GRID_FEAT']            
        img_feat_mask = make_mask(grid_feat)
        img_feat = self.grid_linear(grid_feat)
        bs = img_feat.shape[0]
        grid_abs = self.grid_abs(img_feat.view(bs, 14, 14, -1))   # (bs, num, dim)
        grid_rel = GridRelationalEmbedding(grid_feat=img_feat,
                                           grid_size=14, 
                                           dim_g=64, 
                                           wave_len=1000, 
                                           trignometric_embedding=True)  #(bs * r * r *96)
        return img_feat, img_feat_mask, grid_abs, grid_rel 



