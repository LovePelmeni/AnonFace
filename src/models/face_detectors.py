from facenet_pytorch import MTCNN 
import numpy 
import typing

class ImageFaceDetector(object):

    def __init__(self, 
        image_size: int,
        crop_margin_size: int, 
        min_face_size: int
    ):
        self._detector = MTCNN(
            image_size=image_size,
            margin=crop_margin_size,
            min_face_size=min_face_size,
            thresholds=[85, 90, 90]
        )

    def detect(self, input_img: numpy.ndarray) -> typing.List:

        detected_faces, _ = self._detector.detect(
            img=input_img, 
            landmarks=False
        )
        for idx, box in range(len(detected_faces)):
            detected_faces[idx] = numpy.round(box, decimals=0)
        return detected_faces