# Modeling

This document explains high-level plan of picking
the classifier network for face recognition tasks, based on
the dataset characteristics.

# Backbone

## Receptive Field Characteristics

Backbone Feature Extractor was trained on X dataset, where ROIs (Region Of Interest)
size varies from (x1:x2) to (x2:x4), therefore, the suggested 
size for the backbone varies from N to M pixels (we came up with the size slightly higher, than the maximum shape of the ROI to preserve some details, whichs can theoretically lay on the edge).

<img of the backbone classifier with small subdescription>

## Results 

<img of result feature maps, obtained by grad-cam, as a result of good feature extraction capabilities>

## Anchor boxes configuration

<img of object min max aspect ratios>

ROIs appears at different resolutions starting
from X to Y, while the aspect ratios remains to be similar
across all possible ROI sizes.

Minimum size of the ROI reaches X,
while maximum size reaches Y.

For anchor box sizes we decided to pick minimum
size slightly below the minimum, that rounds up to
power of the 2, which is G. Applied the same strategy 
for the largest possible size.

Example:
    - lowest ROI size is 78x96 -> anchor box min size - (64x64)
    - largest ROI size is 500x495 -> anchor box max size - (512x512)


Final anchor box configuration for the RPN (Region Proposal Network)

- anchor_box_aspect_ratios - (1:1, 2:1)
- anchor_box_sizes - (64:64, 128:128, 256:256, 512:512)

<img of anchor boxes>

## NMS and proposals filtering
We came up with the version of NMS algorithm 
called Soft-NMS. Standard NMS algorithm simply
does eliminate boxes, which overlap and have
high confidence score, which works poorly with the data,
where ROIs appear to be close to each other or overlap. 
Soft-NMS preserves all conflicting predictions.

# Overall Architecture

<img of the overall model architecture>


# References


[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks by Mingxing Tan, Quoc V. Le]("https://arxiv.org/abs/1905.11946")
[Calculating Receptive Fields for Convolutional Neural Networks]("https://distill.pub/2019/computing-receptive-fields")
[Soft-NMS. Improving Object Detection With One Line of Code by Navaneeth Bodla, Bharat Singh, Rama Chellappa, Larry S. Davis]("https://arxiv.org/abs/1704.04503")


