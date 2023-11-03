import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import init

def RegionRelationalEmbedding(boxes, dim_g=64, wave_len=1000, trignometric_embedding=False):
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

    x_min, y_min, x_max, y_max = torch.chunk(boxes, 4, dim=-1)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    wh = w / h
    area = w * h    

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(1, -1))
    delta_h = torch.log(h / h.view(1, -1))
    detla_wh = torch.log(wh / wh.view(1, -1)) 
    delta_area = torch.log(area / area.view(1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(matrix_size[0], matrix_size[1], 1)
    delta_y = delta_y.view(matrix_size[0], matrix_size[1], 1)
    delta_w = delta_w.view(matrix_size[0], matrix_size[1], 1)
    delta_h = delta_h.view(matrix_size[0], matrix_size[1], 1)
    delta_wh= delta_area.view(matrix_size[0], matrix_size[1], 1)
    delta_area = delta_area.view(matrix_size[0], matrix_size[1], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h, delta_wh, delta_area), -1)  # bs * r * r * 6

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8)
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(matrix_size[0], matrix_size[1], 6, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(matrix_size[0], matrix_size[1], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)