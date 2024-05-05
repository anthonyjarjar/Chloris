from models.model import resnet50
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import torch
from PIL import Image 
import torchvision.transforms as transforms 
from fastapi.middleware.cors import CORSMiddleware

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class LocationPredictionRequest(BaseModel):
    longitude: str
    latitude: str
    bird_species: str


app = FastAPI()

model = resnet50()
model.load_state_dict(torch.load("models/resnet_V5.pth", map_location=torch.device('cpu')))
model.eval()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/echo")
async def echo_image():
    return {"string"}

@app.post("/predict")
def read_item(image: UploadFile = File()):
    logging.info("Received prediction request")
    file = image.file
    
    file.seek(0)
    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])

    img = Image.open(file)

    print(img)

    tensor_image = transform(img)


    tensor_image = tensor_image.unsqueeze(0)

    _, predicted = torch.max(model(tensor_image).data, 1)

    print(predicted)

    return predicted.item()

class PredictionRequest(BaseModel):
    longitude: float
    latitude: float
    birdSpecies: str

@app.post("/predict_location")
def predict_location(request_data: PredictionRequest):
    logging.info("Received location prediction request")

    print(request_data)

    longitude, latitude, birdSpecies = request_data

    return {'prediction': 1}
#! uvicorn app:app --host 192.168.1.101 --port 8000

  