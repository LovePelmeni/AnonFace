import unittest
import numpy 
import cv2
from src.inference.predict import InferenceModel
import pathlib

numpy.random.seed(0)

class NetworkInferenceTestCase(unittest.TestCase):

    def setUp(self):

        self.test_config_path = pathlib.Path("./configs/inference_config.json")

        self.model = InferenceModel.from_config(
            config_path=self.test_config_path
        )

        self.test_input = {
            "test_invalid_img": numpy.clip(
                a=numpy.random.random(size=(512, 512, 3)), 
                a_min=0, a_max=255
            ),
            "test_valid_img": cv2.imread(
                "./ml_tests/samples/test_image.png", 
                cv2.IMREAD_UNCHANGED
            )
        }

    def tearDown(self):
        del self.test_input

    def test_invalid_img(self):
        output_img = self.model.remove_face(
            removal_type='blur',
            input_img=self.test_input['test_invalid_img']
        )
        self.assertEqual(
            first=output_img, 
            second=self.test_input['test_valid_img'],
            msg='images should be identical, because not faces should be detected'
        )

    def test_remove_blur(self):

        output_img = self.model.remove_face(
            removal_type='blur', 
            input_img=self.test_input['test_valid_img']
        )

        self.assertNotAlmostEqual(
            first=output_img,
            second=self.test_input['test_valid_img']
        )

    def test_remove_blackout(self):

        output_img = self.model.remove_face(
            removal_type='blackout', 
            input_img=self.test_input['test_valid_img']
        )

        self.assertNotAlmostEqual(
            first=output_img,
            second=self.test_input['test_valid_img']
        )