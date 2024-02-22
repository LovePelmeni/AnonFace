import numpy 
import cv2
from src.face_utils import (
    face_keypoint_detector, 
    face_removers
)

class FaceProcessor(object):
    """
    API Class, that incorporates all face processing
    abstractions into a single object, so you can
    perform face removals on the image by calling a 
    single method

    Parameters:
    -----------
        keypoint_model_path - path to the model for detecting N facial landmarks
    """
    def __init__(self, 
        keypoint_detector: face_keypoint_detector.FaceKeypointDetector,
        face_remover: face_removers.FaceRemover
    ):
        self._face_remover = face_remover
        self.keypoint_detector = keypoint_detector
   
    def remove_image_face(self, input_img: numpy.ndarray):

        face_kp_coords = self.keypoint_detector.find_keypoints(input_face_img=input_img)
        convex_kp_coords = cv2.convexHull(face_kp_coords)

        binary_mask = numpy.zeros(input_img.shape)

        binary_mask = cv2.fillConvexPoly(
            binary_mask, 
            points=convex_kp_coords, 
            color=[255]*len(input_img.shape)
        )

        # image with applied facial removal technique
        removed_mask = self._face_remover.apply_to_face(input_img)

        # filtering out, which regions should be left untouched 
        # and which should be supplanted by the mask (i.e removed
        
        removed_face_img = numpy.where(
            binary_mask == [255]*len(input_img.shape), 
            removed_mask, input_img
        )
        return removed_face_img