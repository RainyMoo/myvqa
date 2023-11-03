import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import init

def RegionAbsolutePosition(boxes, imgs_wh):
    # boxes -- (bs, num_regions, 4), imgs_wh -- (bs, 2) '''
    #x, y, w, h = boxes[:, :, 0], boxes[:, :, 1], boxes[:, :, 2] - boxes[:, :, 0], boxes[:, :, 3] - boxes[:, :, 1]
    cx = (boxes[:, :, 0] + boxes[:, :, 2]) * 0.5
    cy = (boxes[:, :, 1] + boxes[:, :, 3]) * 0.5
    w = (boxes[:, :, 2] - boxes[:, :, 0]) + 1.
    h = (boxes[:, :, 3] - boxes[:, :, 1]) + 1. 
    ratio_wh = w / h
    area = w * h
    expand_wh = torch.cat([imgs_wh, imgs_wh], dim=1).unsqueeze(dim=1)    #(bs, 1, 4)
    ratio_wh = ratio_wh.unsqueeze(dim=-1)  #(bs, num_r, 1)
    ratio_area = area / (imgs_wh[:, 0] * imgs_wh[:, 1]).unsqueeze(-1) #(bs, num_r)
    ratio_area = ratio_area.unsqueeze(-1) #(bs, num_r, 1)
    boxes = torch.stack([cx, cy, w, h], dim=2)
    boxes = boxes / expand_wh   #(bs, num_r, 4)
    res = torch.cat([boxes, ratio_wh, ratio_area], dim=-1)  #(bs, num_r, 6)
    return res

class PositionEncoder(nn.Module):
    '''Relative position Encoder
    '''
    def __init__(self, embed_dim, posi_dim=5):
        super(PositionEncoder, self).__init__()
        self.proj = nn.Linear(posi_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, boxes, imgs_wh):
        # boxes -- (bs, num_regions, 4), imgs_wh -- (bs, num_regions, 2)
        bs, num_regions = boxes.size()[:2]
        posi = absoluteEncode(boxes, imgs_wh)   #(bs, num_r, 4)

        x = self.proj(posi) 
        x = self.sigmoid(x)
        return x

def RegionRelationalEmbedding(boxes, dim_g=64, wave_len=1000, trignometric_embedding=True):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image
    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j
    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    # returns a relational embedding for each pair of bboxes, with dimension = dim_g
    # follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    batch_size = boxes.size(0)
    x_min, y_min, x_max, y_max = torch.chunk(boxes, 4, dim=-1)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    wh = w / h
    area = w * h

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))
    detla_wh = torch.log(wh / wh.view(batch_size, 1, -1)) 
    delta_area = torch.log(area / area.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_wh= delta_area.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_area = delta_area.view(batch_size, matrix_size[1], matrix_size[2], 1)

    #position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h, delta_wh, delta_area), -1)  # bs * r * r * 6
    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h, delta_wh, delta_area), -1)
    
    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 6, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)


def GridRelationalEmbedding(grid_feat, grid_size=8, dim_g=64, wave_len=1000, trignometric_embedding=True):
    # make grid
    batch_size = grid_feat.shape[0]
    a = torch.arange(0, grid_size).float().cuda()
    c1 = a.view(-1, 1).expand(-1, grid_size).contiguous().view(-1)
    c2 = a.view(1, -1).expand(grid_size, -1).contiguous().view(-1)
    c3 = c1 + 1
    c4 = c2 + 1
    f = lambda x: x.view(1, -1, 1).expand(batch_size, -1, -1)
    x_min, y_min, x_max, y_max = f(c1), f(c2), f(c3), f(c4)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    #w = (x_max - x_min) + 1.
    #h = (y_max - y_min) + 1.
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    ratio = w / h
    area = w * h 

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))
    delta_ratio = torch.log(ratio / ratio.view(batch_size, 1, -1))
    delta_area = torch.log(area / area.view(batch_size, 1, -1))
    
    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_ratio = delta_ratio.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_area = delta_area.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h, delta_ratio, delta_area), -1)  # bs * r * r *6

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 6, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)


def get_normalized_grids(bs, grid_size=8):
    a = torch.arange(0, grid_size).float().cuda()
    c1 = a.view(-1, 1).expand(-1, grid_size).contiguous().view(-1)
    c2 = a.view(1, -1).expand(grid_size, -1).contiguous().view(-1)
    c3 = c1 + 1
    c4 = c2 + 1
    f = lambda x: x.view(1, -1, 1).expand(bs, -1, -1) / grid_size
    x_min, y_min, x_max, y_max = f(c1), f(c2), f(c3), f(c4)
    return y_min, x_min, y_max, x_max


def AllRelationalEmbedding(boxes, dim_g=64, wave_len=1000, trignometric_embedding=False, require_all_boxes=False):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image
    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j
    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    # returns a relational embedding for each pair of bboxes, with dimension = dim_g
    # follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    batch_size = boxes.size(0)
    x_min, y_min, x_max, y_max = torch.chunk(boxes, 4, dim=-1)
    grid_x_min, grid_y_min, grid_x_max, grid_y_max = get_normalized_grids(batch_size)

    x_min = torch.cat([x_min, grid_x_min], dim=1)
    y_min = torch.cat([y_min, grid_y_min], dim=1)
    x_max = torch.cat([x_max, grid_x_max], dim=1)
    y_max = torch.cat([y_max, grid_y_max], dim=1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    area = w * h 
    
    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))
    delta_area = torch.log(area / area.view(batch_size, 1, -1))
    
    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_area = delta_area.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h, delta_area), -1)  # bs * r * r *5

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 5, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    if require_all_boxes:
        all_boxes = torch.cat([x_min, y_min, x_max, y_max], dim=-1)
        return (embedding), all_boxes
    return (embedding)

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)
        not_mask = (mask == False)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)  # .permute(0, 3, 1, 2)
        pos = pos.flatten(1, 2)
        return pos