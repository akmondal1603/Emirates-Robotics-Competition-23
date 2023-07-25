#### RF1

import torch
import torchvision

CATEGORIES = ['apex']

device = torch.device('cuda')
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2 * len(CATEGORIES))
model = model.cuda().eval().half()

###

import torch
from torch2trt import TRTModule

model_trt = TRTModule()
model_trt.load_state_dict(torch.load('road_following_model_trt1.pth'))

###

from jetracer.nvidia_racecar import NvidiaRacecar

car = NvidiaRacecar()

###

from jetcam.usb_camera import USBCamera

camera = USBCamera(width=224, height=224, capture_width=640, capture_height=480, capture_device=0)

###

from utils import preprocess
import numpy as np

STEERING_GAIN = 0.9
STEERING_BIAS = 0.00
car.steering_offset = 0.2

car.throttle = 0.195
car.throttle_gain = 1

while True:
    image = camera.read()
    image = preprocess(image).half()
    output = model_trt(image).detach().cpu().numpy().flatten()
    x = float(output[0])
    car.steering = x * STEERING_GAIN + STEERING_BIAS
