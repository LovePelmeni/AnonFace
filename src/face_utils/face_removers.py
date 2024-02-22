import numpy
import abc 
from functools import partial
import cv2

class FaceRemover(abc.ABC):
    """
    Utility for updating human 
    face on the image in the way, that would
    preserve their identity and be unrecognizable
    """
    @abc.abstractmethod
    def apply_to_face(self, **kwargs) -> numpy.ndarray:
        """
        Method, which applies specific operation
        to remove facial details from the image.
        
        Returns:
        -------
        face image, processed with a given operation
        """

class FaceBlur(FaceRemover):
    """
    Removes face details by applying
    blurring algorithm kernel to the area,
    where face is located.
    """
    def _get_blur_function(self, img_size: tuple):

        kernel_size = (max(1, int(img_size[0] * 0.3)), max(1, int(img_size[1] * 0.3)))

        k_height = kernel_size[0] + int(kernel_size[0] % 2 == 0)
        k_width = kernel_size[1] + int(kernel_size[1] % 2 == 0)

        sigmaX = 0.3 * ((k_height - 1) * 0.6 - 1) + 0.9
        sigmaY = 0.3 * ((k_width - 1) * 0.6 - 1) + 0.9

        return partial(
            cv2.GaussianBlur, 
            ksize=(k_height, k_width), 
            sigmaX=sigmaX, 
            sigmaY=sigmaY
        )

    def apply_to_face(self, input_face_img: numpy.ndarray) -> numpy.ndarray:
        blur_func = self._get_blur_function(img_size=input_face_img.shape)
        return blur_func(src=input_face_img)


class FaceBlackout(FaceRemover):
    """
    Removes face details by blacking out
    the face region. 
    """
    def apply_to_face(self, input_face_img: numpy.ndarray) -> numpy.ndarray:
        return numpy.zeros(input_face_img.shape)