from fastapi import UploadFile 
from fastapi import responses 
from src.inference.predict import InferenceModel
import os
from fastapi.responses import Response
import cv2

INFERENCE_CONFIG_PATH = os.environ.get("INFERENCE_CONFIG_PATH")
model = InferenceModel.from_config(config_path=INFERENCE_CONFIG_PATH)

def remove_face_from_image(
    media_type: str, 
    facial_method: str, 
    input_image: UploadFile[...]
):
    """
    controller function, which accepts
    incoming images from the client and
    runs them through face removal process.

    Parameters:
    -----------
        method_type - (str) - type of the face removal method to use
        input_image - UploadFile
    """
    ext = media_type.split("/")[-1]

    processed_img = model.remove_face(
        removal_type=facial_method, 
        input_img=input_image
    )
    
    success, enc_content = cv2.imencode(
        ext="." + ext, 
        img=processed_img
    )

    if not success:
        return responses.JSONResponse(
            status_code=500,
            content={"error": "failed to encode image to bytes"}
        )

    return Response(
        content=enc_content,
        status_code=200,
        media_type=media_type
    )
    
def healthcheck():
    return responses.JSONResponse(status_code=200)

def parse_system_metrics():
    return {}
