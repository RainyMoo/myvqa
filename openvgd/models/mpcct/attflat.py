from openvgd.models.mtcct.modules import FC, MLP
import torch.nn as nn
import torch.nn.functional as F
import torch

class AttFlat_img(nn.Module):
    def __init__(self, __C):
        super(AttFlat_img, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HSIZE,
            mid_size=__C.ATTFLAT_MLP_SIZE,
            out_size=__C.ATTFLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )
        self.linear_merge = nn.Linear(__C.HSIZE * __C.ATTFLAT_GLIMPSES, __C.HSIZE)

    def forward(self, x, x_mask=None):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2), -1e9)
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.ATTFLAT_GLIMPSES):
            att_list.append(torch.sum(att[:, :, i: i + 1] * x, dim=1))
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted
    
class AttFlat_lang(nn.Module):
    def __init__(self, __C):
        super(AttFlat_lang, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HSIZE,
            mid_size=__C.ATTFLAT_MLP_SIZE,
            out_size=__C.ATTFLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HSIZE * __C.ATTFLAT_GLIMPSES,
            __C.HSIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.ATTFLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted
    