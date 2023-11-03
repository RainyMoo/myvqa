# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.MTCCT.mtcct import MTCCT_ED
from openvqa.models.MTCCT.adapter import Adapter

import torch.nn as nn
import torch.nn.functional as F
import torch

# position
# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat_lang(nn.Module):
    def __init__(self, __C):
        super(AttFlat_lang, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.HIDDEN_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C
        
        if self.__C.Lang_Global == 'max':
            self.max_lang = nn.AdaptiveMaxPool2d((1, None)) 
            
        elif self.__C.Lang_Global == 'avg':
            self.avg_lang = nn.AdaptiveAvgPool2d((1, None)) 
            
        elif self.__C.Lang_Global == 'att':
            self.att_lang = AttFlat_lang(__C)


        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter(__C)

        self.backbone = MTCCT_ED(__C)

        # Flatten to vector
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        # Classification layers
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)


    def forward(self, frcn_feat, grid_feat, bbox_feat, w_feat, h_feat, spa_graph, ques_ix, ques_tensor):

        # Pre-process Language Feature
        ori_lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)
        
        # global_lang
        ques_ix_addone = torch.ones(ques_ix.size(0), 1).cuda() # (batch 1)
        ques_ix_addone = ques_ix_addone.long()
        ques_ix_mask = torch.cat((ques_ix, ques_ix_addone), 1) #(batch, 14+1)
        lang_feat_mask = make_mask(ques_ix_mask.unsqueeze(2)) # (batch, 1, 1, 14+1)
        
        # lang_max
        
        # concat(lang_feat, global_max)
        if self.__C.Lang_Global == 'max':
            global_x_in = self.max_lang(lang_feat)
            lang_feat = torch.cat((lang_feat, global_x_in), 1)
                  
        # concat(global_avg, lang_feat)
        elif self.__C.Lang_Global == 'avg':
            global_x_in = self.avg_lang(lang_feat)
            lang_feat = torch.cat((lang_feat, global_x_in), 1)
        
        # concat(global_att, lang_feat)
        elif self.__C.Lang_Global == 'att':
            global_x_in = self.att_lang(lang_feat, ori_lang_feat_mask)
            lang_feat = torch.cat((lang_feat, global_x_in.unsqueeze(1)), 1)
            
        img_feat, img_feat_mask, region_abs, region_rel  = self.adapter(frcn_feat, grid_feat, bbox_feat, w_feat, h_feat, spa_graph)

        # Backbone Framework
        
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask,
            region_abs, 
            region_rel 
        )

        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        # Classification layers
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat

