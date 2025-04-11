from io import BytesIO
from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import uvicorn
import tensorflow as tf 
import requests
import json
from fastapi.middleware.cors import CORSMiddleware




app=FastAPI()
origins=[
    "http://localhost",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

endpoint="http://localhost:8501/v1/models/potatoes_model:predict"

CLASS_NAMES=['Pepper__bell___Bacterial_spot',
 'Pepper__bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

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