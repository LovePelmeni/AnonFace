import torch 
from torch import nn
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
class BackboneNetwork(BackboneWithFPN):
    """
    Custom backbone network
    followed by FPN for better predictive
    capabilities
    """
    def __init__(self, **kwargs):
        super(BackboneNetwork, self).__init__(**kwargs)
class FasterRCNNFaceDetector(nn.Module):
    """
    Detection network for 
    finding human faces on different resolution networks

    Parameters:
    -----------
        - num_classes: int - number of output classes (without background)
        - input_channels: (int) - number of channels in input images
        - backbone (BackboneNetwork) - custom backbone network for feature extraction (optional)
        - anchor_box_configuration (dict) - optional custom configuration for anchor boxes
        - customer_fasterrcnn_weights - pretrained weights for fasterrcnn (optional)
    """
    def __init__(self,
        num_classes: int, 
        input_channels: int,
        backbone: BackboneNetwork = None,
        anchor_box_configuration: dict = None,
        custom_fasterrcnn_weights = None
    ):
        self.input_conv = nn.Conv2d(in_channels=input_channels, out_channels=1024)
        self.detector = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes+1,
            rpn_anchor_generator=self._get_anchor_setup(
                aspect_ratios=anchor_box_configuration.get("aspect_ratios"),
                aspect_ratio_sizes=anchor_box_configuration.get("aspect_ratio_sizes")
            ),
            weights=custom_fasterrcnn_weights
        )

    def _get_anchor_setup(self, 
        aspect_ratios: list = None, 
        aspect_ratio_sizes: list = None
    ) -> None:
        """
        Initializes custom anchor boxes
        for RPN network.
        
        Parameters:
        ----------
            - aspect_ratios - list of aspect ratios of anchor boxes (1:1, 1:2, etc..)
            - aspect_ratio_sizes - list of aspect ratio sizes of anchor boxes (64:64, 128:128, 234:116, etc..)
        """
        if aspect_ratios is None:
            anchor_sizes = ((anchor_size,) for anchor_size in aspect_ratio_sizes)
            anchor_ratios = (tuple(aspect_ratios),)*len(aspect_ratio_sizes)
        else:
            anchor_sizes, anchor_ratios = None, None 
        
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=anchor_ratios
        )
        return anchor_generator

    def forward(self, input_images: torch.Tensor):
        output = self.input_conv(input_images)
        output = self.detector(output)
        return output