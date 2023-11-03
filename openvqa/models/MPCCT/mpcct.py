# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------
# Flowt gate

from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch
import math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class COMP_MHAtt(nn.Module):
    def __init__(self, __C):
        super(COMP_MHAtt, self).__init__()
        self.__C = __C

        #self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        #self.linear_merge = nn.Sequential(nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE * 2), nn.GLU()) 

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        '''
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)
        '''
        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)
        
        v = k
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


class ABS_MHAtt(nn.Module):
    def __init__(self, __C):
        super(ABS_MHAtt, self).__init__()
        self.__C = __C

        #self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        #self.linear_merge = nn.Sequential(nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE * 2), nn.GLU()) 

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask, region_abs):
        n_batches = q.size(0)
        '''
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)
        '''
        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)
        
        v = k
        #q = self.linear_q(q) + region_abs
        
        q = torch.sigmoid(region_abs) * self.linear_q(q)
        #q = self.linear_q(q) 
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
    
# -----------------------------------
# ----Gated Multi-Head Attention ----
# -----------------------------------

class GATED_MHAtt(nn.Module):
    def __init__(self, __C):
        super(GATED_MHAtt, self).__init__()
        self.__C = __C

        #self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        #self.linear_cv = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        #self.linear_ck = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        #self.linear_cq = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_c1 = nn.Linear(__C.HIDDEN_SIZE,1)
        self.linear_c2 = nn.Linear(__C.HIDDEN_SIZE, 1)
        #self.linear_c3 = nn.Linear(__C.HIDDEN_SIZE, 1)
        #self.linear_gv = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.linear_gq = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.linear_gk = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.WGs = nn.ModuleList([nn.Linear(96, 1, bias=True) for _ in range(8)])
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        #self.linear_merge = nn.Sequential(nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE * 2), nn.GLU()) 

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, c, mask, region_abs, region_rel):
        n_batches = q.size(0)
        #v = self.linear_v(v)
        
        '''
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)
        '''
        # 1. gated Q and K  K=(1+alpha)K  Q=(1+belta)Q 
        
        '''
        c1 = torch.sigmoid(self.linear_c1(c))
        c2 = torch.sigmoid(self.linear_c2(c))
        
        k = (1 + c1) * self.linear_k(k)
        
        q = (1 + c2) * self.linear_q(q)
        '''
      
        #3. gated Q and K  K=alpha*K  Q=belta*Q, alpha=sigmoid(matmul(k，lin1(c))), belta=sigmoid(matmul(Q，lin2(c)))
        '''
        k = self.linear_k(k)
        scores_k = torch.matmul(k, self.linear_c1(c).transpose(-2, -1))
        alpha = torch.sigmoid(scores_k)
        k = alpha * k
        
        q = self.linear_q(q)
        scores_q = torch.matmul(q, self.linear_c2(c).transpose(-2, -1))
        belta = torch.sigmoid(scores_q)
        q = belta * q
        '''
               
        # 4. gated Q and K  K=alpha*K  Q=belta*Q, alpha=sigmoid(matmul(k，lin(c))), belta=sigmoid(matmul(Q，lin(c)))
        '''
        c = self.linear_c1(c)
        k = self.linear_k(k)
        scores_k = torch.matmul(k, c.transpose(-2, -1))
        alpha = torch.sigmoid(scores_k)
        k = alpha * k
        
        q = self.linear_q(q)
        scores_q = torch.matmul(q, c.transpose(-2, -1))
        belta = torch.sigmoid(scores_q)
        q = belta * q
        '''
        
        '''
        # 5. sum sigmoid(w1k+w2c), sigmoid(w3q+w4c) w1, w2, w3, w4(512,512)
        k = self.linear_k(k)
        gate_k = self.linear_gk(k) + self.linear_c1(c)
        alpha = torch.sigmoid(gate_k)
        k = alpha * k
        
        q = self.linear_q(q)
        gate_q = self.linear_gq(q) + self.linear_c2(c)
        belta = torch.sigmoid(gate_q)
        q =belta * q
        '''
        
        # 6. sum sigmoid(w1k+w2c), sigmoid(w3q+w4c) w1, w2, w3, w4(512,1)  
        
        k = self.linear_k(k)
        v = k
        gate_k = self.linear_gk(k) + self.linear_c1(c)
        alpha = torch.sigmoid(gate_k)
        k = alpha * k
        
        q = self.linear_q(q)
        gate_q = self.linear_gq(q) + self.linear_c2(c)
        belta = torch.sigmoid(gate_q)
        q =belta * q
        
        #q = q + region_abs
        #k = k + region_abs
        k = torch.sigmoid(region_abs) * k
        q = torch.sigmoid(region_abs) * q
        
        # 7
        '''
        c = c.expand_as(q)
        q = self.linear_q(q)
        gate_q = self.linear_c1(self.linear_cq(c))
        belta = torch.sigmoid(gate_q)
        q = belta * q
        
        k = self.linear_k(k)
        #v = k
        gate_k = self.linear_c2(self.linear_ck(c))
        thelta = torch.sigmoid(gate_k)
        k = thelta * k
                
        v = self.linear_v(v)
        gate_v = self.linear_c3(self.linear_cv(c))
        alpha = torch.sigmoid(gate_v)
        v = alpha * v
        '''
        
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
        
        v = v.view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)
        flatten_relative_geometry_embeddings = region_rel.view(-1, 96)
        box_size_per_head = list(region_rel.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        w_g = F.relu(relative_geometry_weights)
        
        atted = self.att(v, k, q, w_g, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, w_g, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        scores = scores + torch.log(torch.clamp(w_g, min=1e-6))
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
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = COMP_MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()
        
        self.mhatt1 = GATED_MHAtt(__C)
            
        self.mhatt2 = ABS_MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, c, x_mask, y_mask, region_abs, region_rel):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, c=c, mask=x_mask, region_abs=region_abs, region_rel=region_rel)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask, region_abs=region_abs)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------


class MPCCT_ED(nn.Module):
    def __init__(self, __C):
        super(MTCCT_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])
        self.lstm = nn.LSTM(
            input_size=__C.HIDDEN_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

    def forward(self, y, x, y_mask, x_mask, region_abs, region_rel):
        # Get encoder last hidden vector
        c = y[:,-1:,:]
        for enc in self.enc_list:
            y = enc(y, y_mask)
            c = torch.cat((c, y[:,-1:,:]), 1)
        _, (h,_) = self.lstm(c)
        #_, h = self.lstm(c)
        c = h.squeeze(0).unsqueeze(1)
        
        #y = torch.cat([y[:,:-1,:], c], dim=1)
        #c = y[:,-1:,:]
        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            x = dec(x, y, c, x_mask, y_mask, region_abs, region_rel)

        return y, x
