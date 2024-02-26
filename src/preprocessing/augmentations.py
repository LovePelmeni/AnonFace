import albumentations 
from src.preprocessing import resize 
import cv2 

def get_train_augmentations(
    image_height: int, image_width: int, 
    norm_mean: list, norm_std: list
):
    """
    Set of augmentations for training
    face detector.

    Parameters:
    ----------
        image_height: int - height of the output image
        image_width: int - width of the output image
        norm_mean: typing.List[float] - list of means for image normalization
        norm_std: typing.List[float] - list of stds for image normalization
    """
    return albumentations.Compose(
        transforms=[
            albumentations.ImageCompression(quality_lower=80, quality_upper=100),
            albumentations.OneOf([
                albumentations.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                ),
                albumentations.FancyPCA(),
                albumentations.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2
                )
            ]),
            albumentations.OneOf([
                albumentations.HorizontalFlip(),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.1, 
                    rotate_limit=10
                )
            ]),
            albumentations.OneOf([
                resize.IsotropicResize(
                    image_height=image_height,
                    image_width=image_width,
                    interpolation_up=cv2.INTER_LINEAR,
                    interpolation_down=cv2.INTER_NEAREST,
                ),
                resize.IsotropicResize(
                    image_height=image_height,
                    image_width=image_width,
                    interpolation_up=cv2.INTER_NEAREST,
                    interpolation_down=cv2.INTER_LINEAR
                ),
                resize.IsotropicResize(
                    image_height=image_height,
                    image_width=image_width,
                    interpolation_up=cv2.INTER_CUBIC,
                    interpolation_down=cv2.INTER_LINEAR,
                ),
            ]),
            albumentations.PadIfNeeded(
                min_height=image_height,
                min_width=image_width,
                border_mode=cv2.BORDER_CONSTANT
            ) ,
            albumentations.Normalize(
                mean=norm_mean, std=norm_std, 
                always_apply=True
            )
        ],
        bbox_params=albumentations.BboxParams(
            format='coco', 
            label_fields='bboxes'
        )
    )

def get_validation_augmentations(
    image_height: int, image_width: int, 
    norm_mean: list, norm_std: list
):
    return albumentations.Compose(
        transforms=[
            albumentations.OneOf([
                albumentations.HorizontalFlip(),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.1, 
                    rotate_limit=10
                )
            ]),
            albumentations.OneOf([
                resize.IsotropicResize(
                    image_height=image_height,
                    image_width=image_width,
                    interpolation_up=cv2.INTER_LINEAR,
                    interpolation_down=cv2.INTER_NEAREST,
                ),
                resize.IsotropicResize(
                    image_height=image_height,
                    image_width=image_width,
                    interpolation_up=cv2.INTER_NEAREST,
                    interpolation_down=cv2.INTER_LINEAR
                ),
                resize.IsotropicResize(
                    image_height=image_height,
                    image_width=image_width,
                    interpolation_up=cv2.INTER_CUBIC,
                    interpolation_down=cv2.INTER_LINEAR,
                ),
            ]),
            albumentations.PadIfNeeded(
                min_height=image_height,
                min_width=image_width,
                border_mode=cv2.BORDER_CONSTANT
            ) ,
            albumentations.Normalize(mean=norm_mean, std=norm_std)
        ], bbox_params=albumentations.BboxParams(format='coco', label_fields='bboxes')
    )