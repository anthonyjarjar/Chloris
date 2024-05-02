from models.model import resnet18
from fastapi import FastAPI, File, UploadFile

import torch
from PIL import Image 
import torchvision.transforms as transforms 

app = FastAPI()

model = resnet18()
model.load_state_dict(torch.load("models/resnet50_V2.pth", map_location=torch.device('cpu')))
model.eval()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def read_item(image: UploadFile = File()):

    file = image.file
    


    file.seek(0)
    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor()])

    img = Image.open(file)

    print(img)

    tensor_image = transform(img)


    tensor_image = tensor_image.unsqueeze(0)

    _, predicted = torch.max(model(tensor_image).data, 1)

    print(predicted)

    return predicted.item()


  