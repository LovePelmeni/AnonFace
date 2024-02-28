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

    def find_keypoints(self, input_face_img: numpy.ndarray) -> typing.List:
        """
        Returns list of keypoints coordinates,
        detected by the network on the image
        
        Parameters:
        -----------
            input_img - 
            boxes - list of bounding boxes, of human faces
        """
        try:
            faces = self.detector(input_face_img)
            faces_landmarks = []
            img_height, img_width = input_face_img.shape[:2]
            
            for face in faces:
                landmarks = self.shape_predictor(input_face_img, face)
                processed_landmarks = self._process_landmarks(
                    img_height, 
                    img_width, 
                    landmarks
                )
                faces_landmarks.append(processed_landmarks)
            return numpy.asarray(faces_landmarks)
            
        except(Exception) as err:
            logger.error(err)
            return []
