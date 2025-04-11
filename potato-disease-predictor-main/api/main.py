from io import BytesIO
from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import uvicorn
import tensorflow as tf 
# from keras.layers import TFSMLayer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="My API",
        version="2.0",
        description="My API description",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app=FastAPI()
app.openapi = custom_openapi

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# # Load the model using TFSMLayer
# tfs_layer = TFSMLayer("D:/potato-disease/models/1", call_endpoint='serving_default')

# # Create a Keras model that uses the TFSMLayer
# MODEL = tf.keras.Sequential([
#     tfs_layer
# ])

MODEL =tf.keras.models.load_model("D:/potato-disease/models-served/1/1.keras")



CLASS_NAMES=['Early Blight','Healthy','Late Blight']
@app.get("/ping")
async def ping():
    return "Hello"

def read_file_as_image(file) -> np.ndarray:
    image=np.array(Image.open(BytesIO(file)))
    return image


@app.post("/predict")
async def predict(file:UploadFile = File(...) ):
    image=read_file_as_image(await file.read())
    image_batch =np.expand_dims(image,0)
    predicted=MODEL.predict(image_batch)
    predicted_class=CLASS_NAMES[np.argmax(predicted[0])]
    confidence=round(100*(np.max(predicted[0])),2)
    print(predicted_class,confidence)
    return {
        "class":predicted_class,
        "confidence":confidence
    }
    # predicted_class=CLASS_NAMES[np.argmax(predicted['output_0'])]
    # confidence=round(100*(np.max(predicted['output_0'])),2)
    # print(predicted_class,confidence)
    # return {
    #     "class":predicted_class,
    #     "confidence":confidence
    # }
    pass

if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8000)