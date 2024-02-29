import fastapi 
from src.rest import controllers
import os 

DEBUG = os.environ.get("DEBUG", False)
application = fastapi.FastAPI(debug=DEBUG)

application.add_api_route(
    path="/remove/face/details",
    endpoint=controllers.remove_face_from_image
)

