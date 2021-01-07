import cv2
import numpy as np
import random

import torch
from torchvision import transforms

from utils import ReadImage



class Rotate:
    def __init__(self, angles):
        self.angles = angles
    
    def __call__(self, img):
        angle = random.sample(self.angles, 1)[0]
        return transforms.functional.rotate(img, angle)


class MixUp:
    def __call__(self, x1, y1, x2, y2):
        a = random.randint(1, 5) * 0.1
        batchSize = x1.shape[0]
        device = x1.device
        lam = np.random.beta(a, a, batchSize).reshape(batchSize, 1, 1, 1)
        lam = torch.from_numpy(lam).to(device)
        ind = np.random.permutation(batchSize)
        x = lam * x1[ind] + (1 - lam) * x2
        y = lam * y1[ind] + (1 - lam) * y2
        return x, y


class RGBToYCrBb:
    def __init__(self, onlyY=False, resize=None):
        self.onlyY = onlyY
        self.resize = (resize[1], resize[0]) if resize else None
    
    def __call__(self, img):
        if self.onlyY:
            imgY = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0:1]
            if self.resize:
                imgY = cv2.resize(imgY, self.resize, interpolation=cv2.INTER_CUBIC)

            return imgY

        else:
            imgYCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            if self.resize:
                imgYCrCb = cv2.resize(imgYCrCb, self.resize, interpolation=cv2.INTER_CUBIC)

            return imgYCrCb


class YCrCbToRGB:
    def __init__(self, resize=None, originImgPath=''):
        self.resize = (resize[1], resize[0]) if resize else None
        self.originImgPath = originImgPath

    def __call__(self, img):
        if img.shape[-1] == 3:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
            if self.resize:
                imgRGB = cv2.resize(imgRGB, self.resize, interpolation=cv2.INTER_CUBIC)
            
            return imgRGB
        
        elif img.shape[-1] == 1:
            imgY = img
            if not self.originImgPath: 
                raise Exception("In YCrCbToRGB: Need the path of the origin image to convert YCrCb to RGB !")
            
            originImg = ReadImage(self.originImgPath)
            if self.resize:
                originImg = cv2.resize(originImg, self.resize, interpolation=cv2.INTER_CUBIC)

            imgYCrCb = cv2.cvtColor(originImg, cv2.COLOR_RGB2YCrCb)
            imgYCrCb[:, :, 0] = imgY[:, :, 0]
            imgRGB = cv2.cvtColor(imgYCrCb, cv2.COLOR_YCrCb2RGB)
            return imgRGB
        
        else:
            raise Exception(f"In YCrCbToRGB: Incorrect number of channels of the image (shape = {img.shape}) !")
            

            


