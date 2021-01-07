import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import random
import os
import copy
import imagesize

import torch
from torch.utils.data import DataLoader, Dataset


def PlotImageSizeHist(dirs, bins=100, xRight=1000, figsize=(18, 10)):
    widths, heights = [], []
    for dir in dirs:
        paths = glob.glob(os.path.join(dir, '*'))
        for path in paths:
            width, height = imagesize.get(path)
            widths .append(width)
            heights.append(height)
    
    plt.figure(figsize=figsize)
    plt.subplot(211)
    plt.hist(widths,  bins); plt.title("Width") ; plt.xlim([0, xRight]); plt.xticks(range(0, xRight+1, xRight//bins), rotation=60)
    plt.subplot(212)
    plt.hist(heights, bins); plt.title("Height"); plt.xlim([0, xRight]); plt.xticks(range(0, xRight+1, xRight//bins), rotation=60)
    return widths, heights

def ConvertToOriginPath(path):
    dirNow, name = os.path.split(path)
    dirOrigin    = dirNow + "_origin"
    pathOrigin   = os.path.join(dirOrigin, name)
    return pathOrigin


def ReadImage(imgPath):
        img = plt.imread(imgPath)
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        
        return np.clip(img, 0, 1)


def DownAndUp(img, shapeLR, shapeHR, reUpSize):
    img = cv2.resize(img, shapeLR[::-1], interpolation=cv2.INTER_CUBIC)
    if reUpSize:
        img = cv2.resize(img, shapeHR[::-1], interpolation=cv2.INTER_CUBIC)

    return np.clip(img, 0, 1)


def CreateSubImagePath(saveDir, originImgPath, row, col, saveFormat):
    newName = os.path.split(os.path.splitext(originImgPath)[0])[-1] + f"_{row}{col}" + f".{saveFormat}"
    return os.path.join(saveDir, newName)


def CreateSubImages(originImgPaths, saveDirHR, saveDirLR, clip, scale, reUpSize=True, saveHR=True, saveLR=True, saveFormat='png'):
    if not os.path.exists(saveDirHR):
        os.makedirs(saveDirHR)
    
    if not os.path.exists(saveDirLR):
        os.makedirs(saveDirLR)

    imgHRShape = clip  # (w, h)
    imgLRShape = (clip[0] // scale, 
                  clip[1] // scale)
    imgHRPaths = []
    imgLRPaths = []
    for originImgPath in originImgPaths:
        img = ReadImage(originImgPath)
        h, w = img.shape[0], img.shape[1]
        numX = w // clip[0]
        numY = h // clip[1]
        if numX > 0 and numY > 0:
            for i in range(numX):
                for j in range(numY):
                    clipHRImg = img[j * clip[1]: (j + 1) * clip[1], i * clip[0]: (i + 1) * clip[0]]
                    clipLRImg = DownAndUp(clipHRImg, imgLRShape, imgHRShape, reUpSize)
                    imgHRPath = CreateSubImagePath(saveDirHR, originImgPath, j, i, saveFormat)
                    imgLRPath = CreateSubImagePath(saveDirLR, originImgPath, j, i, saveFormat)
                    imgHRPaths.append(imgHRPath)
                    imgLRPaths.append(imgLRPath)

                    if saveHR: plt.imsave(imgHRPath, clipHRImg)
                    if saveLR: plt.imsave(imgLRPath, clipLRImg)

    return imgHRPaths, imgLRPaths


def GPUToNumpy(tensor, reduceDim=None):
    if reduceDim is not None:
        return tensor.squeeze(reduceDim).cpu().detach().numpy().transpose(1, 2, 0)
    else:
        return tensor.squeeze(         ).cpu().detach().numpy().transpose(1, 2, 0)


def PlotHRLR(imgHR, imgLR, figsize=(8, 8), plotResidual=False):
    def Enhance(img):
        maxValue   = np.max(img)
        imgEnhance = img / maxValue
        return imgEnhance

    try:
        imgLR    = np.clip(GPUToNumpy(imgLR, reduceDim=0) if type(imgLR) is torch.Tensor else imgLR, 0, 1)
        imgHR    = np.clip(GPUToNumpy(imgHR, reduceDim=0) if type(imgHR) is torch.Tensor else imgHR, 0, 1)
        residual = np.clip(Enhance(imgHR - imgLR), 0, 1 ) if plotResidual else None

        if plotResidual:
            plt.figure(figsize=figsize)
            plt.subplot(131); plt.imshow(imgLR)   ; plt.title("LR image")
            plt.subplot(132); plt.imshow(imgHR)   ; plt.title("HR image")
            plt.subplot(133); plt.imshow(residual); plt.title("Residual")
        else:
            plt.figure(figsize=figsize)
            plt.subplot(121); plt.imshow(imgLR)   ; plt.title("LR image")
            plt.subplot(122); plt.imshow(imgHR)   ; plt.title("HR image")
        
        plt.show()
    
    except Exception as e:
        imgRe = Enhance(imgHR - imgLR)
        print(f"Error in PlotHRLR:\n  {e}")
        print(f"HR image :\n{imgHR}\n[shape={imgHR.shape}]")
        print(f"LR image :\n{imgLR}\n[shape={imgLR.shape}]")
        print(f"Residual :\n{imgRe}\n[shape={imgRe.shape}]")


class ImageDataset(Dataset):
    def __init__(self, imgDir, transform=None):
        self.imgPaths  = glob.glob(os.path.join(imgDir, '*'))
        self.transform = transform
        self.n = len(self.imgPaths)
    
    def __getitem__(self, index):
        imgPath = self.imgPaths[index]
        img = ReadImage(imgPath)
        if self.transform:
            img = self.transform(img)
        
        return img
    
    def __len__(self):
        return self.n
    
    def GetImageShape(self):
        return list(self[0].shape)


class ImageLoader:
    def __init__(self, datasetLR, datasetHR, batchSize):
        if len(datasetLR) != len(datasetHR):
            raise Exception("len(datasetLR) must equal to len(datasetHR) !")

        self.datasetLR = datasetLR
        self.datasetHR = datasetHR
        self.batchSize = batchSize
        self.imgShape  = datasetHR.GetImageShape()
        self.n = len(datasetLR)
        self.allIndex = set(range(self.n))
        self.nowIndex = copy.copy(self.allIndex)
    
    def ChooseBatchIndex(self):
        if len(self.nowIndex) < self.batchSize:
            numPadding     = self.batchSize - len(self.nowIndex)
            paddingChoices = self.allIndex - self.nowIndex
            paddingIndex   = random.sample(paddingChoices, k=numPadding)
            batchIndex     = list(self.nowIndex) + paddingIndex
            self.nowIndex  = copy.copy(self.allIndex)
        else:
            batchIndex = random.sample(self.nowIndex, k=self.batchSize)
            self.nowIndex -= set(batchIndex)
        
        return batchIndex
    
    def __len__(self):
        return self.n
    
    def __iter__(self):
        return self
    
    def __next__(self):
        batchIndex = self.ChooseBatchIndex()
        random.shuffle(batchIndex)
        imgLRs = torch.zeros([self.batchSize, *self.imgShape])
        imgHRs = torch.zeros([self.batchSize, *self.imgShape])
        for i, j in enumerate(batchIndex):
            seed = random.randint(0, 2147483648)

            random.seed(seed); torch.manual_seed(seed)
            imgHRs[i] = self.datasetHR[j]

            random.seed(seed); torch.manual_seed(seed)
            imgLRs[i] = self.datasetLR[j]
        
        
        return (imgLRs, imgHRs)


def ComparePSNR(imgSR_Y, imgHR_Y):
    ySR  = imgSR_Y * 255.
    yHR  = imgHR_Y * 255.
    mse  = np.mean((ySR - yHR) ** 2.)
    psnr = 20 * np.log10(255. / mse)
    return psnr