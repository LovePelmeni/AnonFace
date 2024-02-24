from src.face_utils import face_processing
from src.models import face_detectors
import typing
import logging 

logger = logging.getLogger("inference_logger")
handler = logging.FileHandler(filename="inference_model_logs.log")
logger.addHandler(handler)

class InferenceModel(object):
    
    def __init__(self, 
        face_detector_weights_path: str, 
        face_landmarks_detector_weights_path: str,
        total_face_landmarks: int,
        face_image_size: int,
        face_config: typing.Dict[str, int]
    ):
        self._face_detector = face_detectors.ImageFaceDetector(**face_config)
        self.face_processor = face_processing.FaceProcessor(
            keypoint_model_path=face_landmarks_detector_weights_path, 
            total_model_landmarks=total_face_landmarks
        )