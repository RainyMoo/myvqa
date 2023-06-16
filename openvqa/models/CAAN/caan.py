# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch
import math

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask, c):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

# ------------------------------
# ---- Global_Context MHA ----
# ------------------------------

class GMHAtt(nn.Module):
    def __init__(self, __C):
        super(GMHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.avgpool_k = nn.AdaptiveAvgPool2d((1,None))
        self.avgpool_q = nn.AdaptiveAvgPool2d((1,None))
        self.lin_uk = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.lin_uq = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.tran_q = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.tran_k = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.tran_cq = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.tran_ck = nn.Linear(__C.HIDDEN_SIZE, 1)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask, c=None):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        c_k = self.lin_uk(self.avgpool_k(k)) # (B, 1, 512)
        c_q = self.lin_uq(self.avgpool_q(q))

        k = self.linear_k(k)  # (B, N, 512)
        q = self.linear_q(q)

        merge_q = self.tran_q(q) + self.tran_cq(c_q) # (B. N, 1)
        lamta_q = torch.sigmoid(merge_q)

        merge_k = self.tran_k(k) + self.tran_ck(c_k)
        lamta_k = torch.sigmoid(merge_k)

        q = (1-lamta_q) * q + lamta_q * c_q # (B, N, 512)
        k = (1-lamta_k) * k + lamta_k * c_k

        k = k.view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = q.view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)
# ------------------------------
# ---- Deep-context MHA ----
# ------------------------------

class DMHAtt(nn.Module):
    def __init__(self, __C, l_layer):
        super(DMHAtt, self).__init__()
        self.__C = __C

        self.lin_layer = nn.Linear(__C.HIDDEN_SIZE * l_layer, __C.HIDDEN_SIZE)

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.lin_uk = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.lin_uq = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.tran_q = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.tran_k = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.tran_cq = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.tran_ck = nn.Linear(__C.HIDDEN_SIZE, 1)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask, c):
        n_batches = q.size(0)

        if c.__len__() == 1:
            c = c[0]
        else:
            c = torch.cat(c, dim=-1)
        c = self.lin_layer(c) # (B, N, 512)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        c_k = self.lin_uk(c) # (B, N, 512)
        c_q = self.lin_uq(c)

        k = self.linear_k(k)  # (B, N, 512)
        q = self.linear_q(q)

        merge_q = self.tran_q(q) + self.tran_cq(c_q) # (B. N, 1)
        lamta_q = torch.sigmoid(merge_q)

        merge_k = self.tran_k(k) + self.tran_ck(c_k)
        lamta_k = torch.sigmoid(merge_k)

        q = (1-lamta_q) * q + lamta_q * c_q # (B, N, 512)
        k = (1-lamta_k) * k + lamta_k * c_k

        k = k.view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = q.view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

# ------------------------------
# ---- DeepGlobal-context MHA ----
# ------------------------------

class DGMHAtt(nn.Module):
    def __init__(self, __C, l_layer):
        super(DGMHAtt, self).__init__()
        self.__C = __C

        self.lin_layer = nn.Linear(__C.HIDDEN_SIZE * l_layer, __C.HIDDEN_SIZE)

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.lin_uk = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.lin_uq = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.tran_q = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.tran_k = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.tran_cq = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.tran_ck = nn.Linear(__C.HIDDEN_SIZE, 1)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask, c):
        n_batches = q.size(0)

        if c.__len__() == 1:
            c = c[0]
        else:
            c = torch.cat(c, dim=-1)
        c = self.lin_layer(c) # (B, 1, 512)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        c_k = self.lin_uk(c) # (B, 1, 512)
        c_q = self.lin_uq(c)

        k = self.linear_k(k)  # (B, 1, 512)
        q = self.linear_q(q)

        merge_q = self.tran_q(q) + self.tran_cq(c_q) # (B. N, 1)
        lamta_q = torch.sigmoid(merge_q)

        merge_k = self.tran_k(k) + self.tran_ck(c_k)
        lamta_k = torch.sigmoid(merge_k)

        q = (1-lamta_q) * q + lamta_q * c_q # (B, N, 512)
        k = (1-lamta_k) * k + lamta_k * c_k

        k = k.view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = q.view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)
# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C, l_layer):
        super(SA, self).__init__()
        self.__C = __C
        
        if self.__C.USE_CONTEXT == 'None':
            self.mhatt = MHAtt(__C)
        
        elif self.__C.USE_CONTEXT == 'global':
            self.mhatt = GMHAtt(__C)
            
        elif self.__C.USE_CONTEXT == 'deep':
            self.mhatt = DMHAtt(__C, l_layer = l_layer)
            
        elif self.__C.USE_CONTEXT == 'deep-global':
            self.mhatt = DGMHAtt(__C, l_layer = l_layer)
            
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, y_mask, c):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask, c)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C, l_layer):
        super(SGA, self).__init__()

         #self.mhatt1 = MHAtt(__C)
        self.mhatt1 = DMHAtt(__C, l_layer = l_layer)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask, c):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask, c=c)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask, c=None)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class CAAN_ED(nn.Module):
    def __init__(self, __C):
        super(CAAN_ED, self).__init__()       
        self.__C = __C

        self.enc_list = nn.ModuleList([SA(__C, l_layer = i+1) for i in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C, l_layer = i+1) for i in range(__C.LAYER)])
        
        if __C.USE_CONTEXT == 'deep-global':
            self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        

    def forward(self, y, x, y_mask, x_mask):
        # Get encoder last hidden vector
        
        if self.__C.USE_CONTEXT == 'None':
            for enc in self.enc_list:
                y = enc(y, y_mask, c=None)
        
        elif self.__C.USE_CONTEXT == 'deep':
            c = [y]
            for enc in self.enc_list:
                y = enc(y, y_mask, c)
                c.append(y)
                
        elif self.__C.USE_CONTEXT == 'global':
            for enc in self.enc_list:
                y = enc(y, y_mask, c=None)
                
        elif self.__C.USE_CONTEXT == 'deep-global':
            # use other methods can replace avgpool. 
            c = [self.avgpool(y)]
            for enc in self.enc_list:
                y = enc(y, y_mask, c)
                c.append(self.avgpool(y))

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        c = [x]
        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask, c)
            c.append(x)

        return y, x
