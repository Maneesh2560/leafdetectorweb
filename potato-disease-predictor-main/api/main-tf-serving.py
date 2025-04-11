from io import BytesIO
from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import uvicorn
import tensorflow as tf 
import requests
import json
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

endpoint="http://localhost:8501/v1/models/potatoes_model:predict"

CLASS_NAMES=['Early Blight','Healthy','Late Blight']
headers = {"content-type": "application/json"}

@app.get("/ping")
async def ping():
    return "Hello"

def read_file_as_image(file) -> np.ndarray:
    image=np.array(Image.open(BytesIO(file)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("Received file for prediction")
    try:
        image = read_file_as_image(await file.read())
        print("Image read successfully")
        image_batch = np.expand_dims(image, 0)
        json_data = {'instances': image_batch.tolist()}
        response = requests.post(endpoint, json=json_data,headers=headers)
        prediction = json.loads(response.text)['predictions'][0]
        
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = round(100 * np.max(prediction), 2)
        print(predicted_class,confidence)
        return {"class": predicted_class, "confidence": confidence}
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
    

if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8000)