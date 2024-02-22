import dlib
import numpy 
import typing 
import logging 

logger = logging.getLogger("keypoint_logger")
file_handler = logging.FileHandler(filename="face_keypoints_logs.log")
logger.addHandler(file_handler)
class FaceKeypointDetector(object):
    """
    Detects 68 landmark keypoints on human
    face.

    Parameters:
    -----------
        - keypoint_model_path - path to the .dat file of pretrained landmark model
    """
    def __init__(self, keypoint_model_path: str, total_landmarks: int):
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(keypoint_model_path)
        self.total_landmarks: int = total_landmarks
    
    def _process_landmarks(self, img_height: int, img_width: int, landmarks: typing.List):

        points = []

        for idx in range(0, self.total_landmarks):

            point = [
                min(max(0, landmarks.part(idx).x), img_width-1), 
                min(max(0, landmarks.part(idx).y), img_height-1)
            ]
            points.append(point)
        return points
