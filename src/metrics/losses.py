from torch import nn
from torchvision.ops.diou_loss import distance_box_iou_loss

class DistanceIOULoss(nn.Module):
    """
    Distance-based Intersection Over Union
    Loss function, designed for considering
    distance between bounding boxes, when comparing
    against each other
    """
    def __init__(self):
        super(DistanceIOULoss, self).__init__()

    def forward(self, true_boxes: list, pred_boxes: list):
        return distance_box_iou_loss(
            boxes1=true_boxes,
            boxes2=pred_boxes,
            reduction="mean",
            eps=1e-5
        )
    