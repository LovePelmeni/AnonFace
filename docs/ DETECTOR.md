# Modeling

This document strives to describe the essential parts of the modeling stage
of the project. It highlights subtleties and caveats of the architecture design.
Low Technical concepts are beyong scope of this document. For that reason, all 
relevant material was provided to gain a better understanding of technical choices
made.

# Dataset
We used Celebra Dataset (well-known collections of human faces with other 200.000 images and 40 binary attributes) as a fine-tuning source of data. For more information, visit `docs/DATASET.md`.

## Backbone for Feature extraction

Backbone network plays essential part in object detection networks, as it provides a way of extracting context from the image, before embarking on the main analysis. Having sufficiently deep backbone to capture ROIs at different resolutions is a critical step towards building a good detection network.

## Receptive Field Characteristics

Let's explore some of the common properties of training and validation data splits,
used for training and validation of the face detector.

Training split contains roughly 4000 images with the following characteristics:

    - minimal aspect ratio of the box: 0.46
    - maximum aspect ratio of teh box: 0.87

    - minimal width and height: 53, 115
    - maximum width and height: 346, 398

Validation split contains roughly 900 image with the following characteristics:

    - minimal aspect ratio of the box: 0.51
    - maximum aspect ratio of the box: 0.91

    - minimal width and height: 212, 407
    - maximum width and height: 315, 346

With considerations of this information, our best receptive field should
approximately equal 600x600.

## Solution 

We used EfficientNet-B4 pretrained classifier.

<img-of-backbone-classifier>

## Results 

Here are results, obtained by ("Grad-CAM")["https://arxiv.org/abs/1610.02391"], which demonstrates the network's feature extraction capabilities on Celebra Dataset.

<img of result feature maps, obtained by grad-cam, as a result of good feature extraction capabilities>

Here, detected regions can be seen as an areas with the colors, 
going from blue to red.

## Anchor boxes configuration

<img of object min max aspect ratios>

ROIs appears at different resolutions
from 16:16 to 512:512, 
while the aspect ratios ranges from 1:1, to 1:2.

Final anchor box configuration for the RPN (Region Proposal Network)

- anchor_box_aspect_ratios - [1:1, 1:2]
- anchor_box_sizes - [64:64, 128:128, 256:256, 512:512]

## NMS and proposals filtering
We came up with the version of NMS algorithm 
called Soft-NMS. Standard NMS algorithm has one negative propertty, which may not be suitable for all datasets.

Problem lies in eliminating conflicting proposed box candidates, which overlap and have
high confidence score, which works poorly with the data,
where ROIs appear to be close to each other or overlap. 

Soft-NMS demotes scores of conflicting boxes by some factor, while preserving the box with the highest score among them

# Architecture diagram

<img of the overall model architecture>

# References

[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization by ]("https://arxiv.org/abs/1610.02391")

[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks by Mingxing Tan, Quoc V. Le]("https://arxiv.org/abs/1905.11946")

[Calculating Receptive Fields for Convolutional Neural Networks]("https://distill.pub/2019/computing-receptive-fields")

[Soft-NMS. Improving Object Detection With One Line of Code by Navaneeth Bodla, Bharat Singh, Rama Chellappa, Larry S. Davis]("https://arxiv.org/abs/1704.04503")



