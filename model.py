import numpy as np
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # To get the type of the device the model in being run on

model = EfficientNet.from_pretrained(model_name="efficientnet-b0")  # Get Architecture/Weights of EfficientNet-B0


class EmotionModel(nn.Module):
    """
    Model Class

    Arguments:
        model: The efficientnet model
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        return self.softmax(x)


model = EmotionModel(model)  # Instantiate the Emotion Model Class
in_features = model.model._fc.in_features  # Get the Default In-Features
model.model._fc = nn.Linear(in_features=in_features, out_features=7)  # Replace the FC layer with our own layer
model.load_state_dict(torch.load("models/Best_Loss.pt", map_location=DEVICE))  # Load the pretrained weights for model

model.eval()  # Model to Evaluation Mode, were using this script for prediction here


def predict_emotion(data=None):
    """

    :param
        data: Pixels of Images to Classify
    :return: dict
    """
    model.eval()  # Model to Evaluation Mode
    outputs = model(data)  # Make prediction on data
    outputs = outputs.detach().cpu().numpy()  # Detach and convert to  numpy array
    outputs = np.round(outputs, 0)  # Round values to 2 decimal places
    outputs = outputs.flatten().tolist()  # Flatten out array and convert to a list
    out_map = {
        "Angry": outputs[0] * 100,
        "Disgust": outputs[1] * 100,
        "Fear": outputs[2] * 100,
        "Happy": outputs[3] * 100,
        "Sad": outputs[4] * 100,
        "Neutral": outputs[5] * 100,
        "Surprise": outputs[6] * 100
    }
    return out_map
