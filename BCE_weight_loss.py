import torch
import torch.nn as nn

def BCE_weight_LOSS(pred, target, count_pos, count_neg):
    """
    BCE_weight_LOSS
    The weight depends on the proportion of changing and unchanged samples in each batch
    no-change，0
    change，1
    """
    m = nn.Sigmoid()
    lossinput = m(pred)
    ratio = count_neg / (count_pos + count_neg)
    output = - (ratio * target * torch.log(lossinput + 1e-10) + (1 - ratio) * (1 - target) * torch.log(
        1 - lossinput + 1e-10))
    output = torch.mean(output)
    return output