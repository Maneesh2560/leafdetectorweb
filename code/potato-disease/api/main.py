from io import BytesIO
from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import uvicorn
import tensorflow as tf 
# from keras.layers import TFSMLayer
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

# # Load the model using TFSMLayer
# tfs_layer = TFSMLayer("D:/potato-disease/models/1", call_endpoint='serving_default')

# # Create a Keras model that uses the TFSMLayer
# MODEL = tf.keras.Sequential([
#     tfs_layer
# ])

MODEL =tf.keras.models.load_model("C:/code/potato-disease/models/2")



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