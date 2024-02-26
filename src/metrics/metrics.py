from signal import SIG_DFL
from torch import nn
import torch
class IOUScore(nn.Module):
    """
    Intersection Over Union Score.
    Measures overlap between boxes,
    represented as a set of coordinates
    """
    def __init__(self, eps: float = 1e-7): 
        super(IOUScore, self).__init__()
        self.eps = eps 

    def forward(self, pred_boxes, act_boxes):

        min_x1 = torch.max(pred_boxes[:, 0], act_boxes[:, 0])
        min_x2 = torch.min(pred_boxes[:, 1], act_boxes[:, 1])
        max_y1 = torch.max(pred_boxes[:, 2], act_boxes[:, 2])
        max_y2 = torch.min(pred_boxes[:, 3], act_boxes[:, 3])

        intersection = (min_x2 - min_x1) * (max_y2 - max_y1)
        union = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (act_boxes[:, 3] - act_boxes[:, 1])

        return intersection / union

class AveragePrecision(nn.Module):

class AverageRecall(nn.Module):
