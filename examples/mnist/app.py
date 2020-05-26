from flask import Flask, render_template, request
# from models import MobileNet
from model import Net
import os
from math import floor

import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# from torchvision import datasets, transforms
from torchvision import transforms
import numpy as np

from PIL import Image


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'

########################### START - model stuff ###########################

use_cuda = False; #not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(1); #args.seed
device = torch.device("cuda" if use_cuda else "cpu");
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {};

lr = 0.01;
momentum = 0.5;

model = Net().to(device);
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum);

model_path = "model.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

########################### END - model stuff ###########################


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

def pil_loader(path):
    """Load images from /eval/ subfolder, convert to greyscale and resized it as squared"""
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            sqrWidth = np.ceil(np.sqrt(img.size[0]*img.size[1])).astype(int)
            return img.convert('L').resize((sqrWidth, sqrWidth))

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        saveLocation = f.filename
        f.save(saveLocation)

        image_path = saveLocation

        # input_image = Image.open(image_path)
        input_image = pil_loader(image_path)
        preprocess = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        input_tensor = preprocess(input_image)
        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)
        #output = self.model(input_batch) 

        # input = Variable(input_batch)
        # input = input.to(device)

        #inference, confidence = model.infer(saveLocation)

        # single_loaded_img = test_loader.dataset.data[test_data_loader_index]
        # #print(single_loaded_img)
        # single_loaded_img = single_loaded_img.to(device)
        # single_loaded_img = single_loaded_img[None, None]
        # #single_loaded_img = single_loaded_img.type('torch.DoubleTensor')
        # single_loaded_img = single_loaded_img.type('torch.FloatTensor')

        #raw_prediction = model(input)
        raw_prediction = model(input_batch)

        #maxed_prediction = raw_prediction.max(1, keepdim=True)[1]
        maxed_prediction = raw_prediction.argmax(dim=1, keepdim=True).item()
        # maxed_prediction = maxed_prediction.item()
        print("maxed_prediction:")
        print(maxed_prediction)
        print("")


        output = "hello world"
        return str(maxed_prediction)


# from torchvision import transforms
# class ConvNet
    # def infer(self, image_path):
    #     input_image = Image.open(image_path)
    #     preprocess = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #     input_tensor = preprocess(input_image)

    #     # create a mini-batch as expected by the model
    #     input_batch = input_tensor.unsqueeze(0) 

    #     # move the input and model to GPU for speed if available
    #     if torch.cuda.is_available():
    #         input_batch = input_batch.to('cuda')
    #         self.model.to('cuda')

    #     with torch.no_grad():
    #         output = self.model(input_batch)

    #     # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    #     output = torch.nn.functional.softmax(output[0], dim=0)
    #     confidence, index = torch.max(output, 0)

    #     return (self.classes[index.item()], confidence.item())


# @app.route('/infer', methods=['POST'])
# def success():
#     if request.method == 'POST':
#         f = request.files['file']
#         saveLocation = f.filename
#         f.save(saveLocation)
#         inference, confidence = model.infer(saveLocation)
#         # make a percentage with 2 decimal points
#         confidence = floor(confidence * 10000) / 100
#         # delete file after making an inference
#         os.remove(saveLocation)
#         # respond with the inference
#         return render_template('inference.html', name=inference, confidence=confidence)


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 80))
    app.run(host='0.0.0.0', port=port, debug=True)
