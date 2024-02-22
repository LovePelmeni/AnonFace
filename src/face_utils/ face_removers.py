import numpy 
import typing 
import abc 

class FaceRemover(abc.ABC):
    """
    Utility for updating human 
    face on the image in the way, that would
    preserve their identity and be unrecognizable
    """
    @abc.abstractmethod
    def apply_to_face(self, **kwargs) -> numpy.ndarray:
        pass

class FaceBlur(object):
    """
    Removes face details by applying
    blurring algorithm kernel to the area,
    where face is located.

    Parameters:
    ----------
        - blur_method - typing.Literal - type of the blur method
        - 
    """
    def _get_blur_method(self, blur_method: str):
        pass 

    def apply_to_face(
        input_face_img: numpy.ndarray,
        blur_method: typing.Literal['gaussian', 'median', 'bilateral'] = 'gaussian'
    ) -> numpy.ndarray:
        pass

class FaceBlackout(object):
    """
    Removes face details by blacking out
    the face region. 
    """
    def apply_to_face(self, input_face_img: numpy.ndarray):
        pass