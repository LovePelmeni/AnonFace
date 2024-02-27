from src.face_utils import face_processing
from src.face_utils import face_keypoint_detector
from src.face_utils import face_removers
import typing
import numpy
import logging 
from facenet_pytorch import MTCNN
import json
import glob
import os
import pathlib
import cv2

logger = logging.getLogger("inference_logger")
handler = logging.FileHandler(filename="inference_model_logs.log")
logger.addHandler(handler)

class InferenceModel(object):
    
    @classmethod
    def from_config(cls, config_path: str):
        """
        Setup function to prepare
        inference pipeline.

        Parameters:
        -----------
            config_path - path to .JSON configuration file, containing
            pipeline properties
        """
        if not os.path.exists(config_path):
            root_dir = pathlib.Path(config_path).parent

            config_paths = glob.glob(
                pathname='*.json', 
                root_dir=root_dir, 
                recursive=True
            )

            if not len(config_paths):
                raise FileNotFoundError("Failed to find configuration path. Check validity of the path")

            config_path = os.path.join(root_dir, config_paths[-1])

        configuration = json.load(fp=open(config_path, mode='rb'))

        image_size = configuration.get("input_image_size")
        face_margin_size = configuration.get("face_margin_size", 50)
        min_face_size = configuration.get("min_face_size", 20)

        keypoint_model_path = configuration.get("keypoint_model_path")
        keypoint_landmarks_num = configuration.get("keypoint_total_landmarks")

        cls._detector = MTCNN(
            image_size=image_size,
            margin=face_margin_size,
            post_process=False,
            min_face_size=min_face_size
        )

        keypoint_detector = face_keypoint_detector.FaceKeypointDetector(
            keypoint_model_path=keypoint_model_path,
            total_landmarks=keypoint_landmarks_num
        )

        cls._face_blur_processor = face_processing.FaceProcessor(
            keypoint_detector=keypoint_detector,
            face_remover=face_removers.FaceBlur()
        )
        cls._face_blackout_processor = face_processing.FaceProcessor(
            keypoint_detector=keypoint_detector,
            face_remover=face_removers.FaceBlackout()
        )
        return cls()

    def remove_face(self, 
        removal_type: typing.Literal['blackout', 'blur'], 
        input_img: numpy.ndarray
    ):
        """
        Removes all human faces from the given image.
        
        Parameters:
        -----------
            - removal_type - type of strategy to use for face removal 
            - input_img - input numpy.ndarray image containing human faces.
        
        Returns:
            - new image with removed or blurred faces
        """
        boxes, _ = self._detector.detect(
            input_img, 
            landmarks=False
        )
        
        for box in boxes:

            x1 = int(min(max(0, box[0]), input_img.shape[1]-1))
            y1 = int(min(max(0, box[1]), input_img.shape[1]-1))
            x2 = int(min(max(0, box[2]), input_img.shape[0]-1))
            y2 = int(min(max(0, box[3]), input_img.shape[0]-1))

            face = input_img[y1:y2, x1:x2]
            
            if removal_type == 'blackout':
                removed_face = self._face_blackout_processor.remove_image_face(input_img=face)

            if removal_type == 'blur':
                removed_face = self._face_blur_processor.remove_image_face(input_img=face)

            input_img[y1:y2, x1:x2] = removed_face
        return input_img

