import os
import glob
import copy
import imagesize
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import join
from PIL import Image
from pprint import pprint

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import (Compose, RandomCrop, ToTensor, ToPILImage, Resize,
                                    RandomHorizontalFlip, RandomVerticalFlip)

mean = [0.4488, 0.4371, 0.4040]
std  = [1., 1., 1.]

class Rotate:
    def __init__(self, angles):
        self.angles = angles
    
    def __call__(self, imgs):
        angle = random.sample(self.angles, 1)[0]
        rotated_images = transforms.functional.rotate(imgs, angle)
        return rotated_images


class MixUp:
    def __call__(self, x1, y1, x2, y2):
        a = random.randint(1, 5) * 0.1
        batch_size = x1.shape[0]
        device = x1.device
        lam = np.random.beta(a, a, batch_size).reshape(batch_size, 1, 1, 1)
        lam = torch.from_numpy(lam).to(device)
        ind = np.random.permutation(batch_size)
        x = lam * x1[ind] + (1 - lam) * x2
        y = lam * y1[ind] + (1 - lam) * y2
        return x, y


def show_too_small_image(img_dir, size=96):
    paths = glob.glob(os.path.join(img_dir, '*.*'))
    collect = []
    for path in paths:
        img_name = os.path.basename(path)
        w, h = imagesize.get(path)
        if w < size or h < size:
            collect.append(img_name)
    
    return collect


def gpu_to_numpy(tensor, reduceDim=None, is_squeeze=True):
    if type(tensor) is np.array or type(tensor) is np.ndarray:
        return tensor
    
    if is_squeeze:
        if reduceDim is not None:
            return tensor.squeeze(reduceDim).cpu().detach().numpy().transpose(1, 2, 0)
        else:
            return tensor.squeeze(         ).cpu().detach().numpy().transpose(1, 2, 0)
    
    else:
        if len(tensor.shape) == 3:
            return tensor.cpu().detach().numpy().transpose(1, 2, 0)
        elif len(tensor.shape) == 4:
            return tensor.cpu().detach().numpy().transpose(0, 2, 3, 1)
            


def plot_hr_lr(imgHR, imgLR, figsize=(18, 18), plotResidual=False):
    def enhance(img):
        maxValue   = np.max(img)
        imgEnhance = img / maxValue
        return imgEnhance

    try:
        imgLR    = np.clip(gpu_to_numpy(imgLR, reduceDim=0) if type(imgLR) is torch.Tensor else imgLR, 0, 1)
        imgHR    = np.clip(gpu_to_numpy(imgHR, reduceDim=0) if type(imgHR) is torch.Tensor else imgHR, 0, 1)
        residual = np.clip(enhance(imgHR - imgLR), 0, 1 ) if plotResidual else None

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
        imgRe = enhance(imgHR - imgLR)
        print(f"Error in PlotHRLR:\n  {e}")
        print(f"HR image :\n{imgHR}\n[shape={imgHR.shape}]")
        print(f"LR image :\n{imgLR}\n[shape={imgLR.shape}]")
        print(f"Residual :\n{imgRe}\n[shape={imgRe.shape}]")

    plt.show()
    

def plot_scheduler(scheduler, n_iter=2000, title="LR Scheduler"):
    scheduler = copy.deepcopy(scheduler)
    scheduler.verbose = False
    for i in range(n_iter):
        lr = scheduler.get_last_lr()
        plt.plot(i, lr, 'bo')
        scheduler.step()
    
    plt.title(title)
    plt.show()
    scheduler.last_epoch = -1


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def normalize(imgs, mean=mean, std=std):
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(imgs.device)
    std  = torch.tensor(std ).view(1, 3, 1, 1).to(imgs.device)
    return (imgs - mean) / std


def denormalize(imgs, mean=mean, std=std):
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(imgs.device)
    std  = torch.tensor(std ).view(1, 3, 1, 1).to(imgs.device)
    return (imgs * std) + mean


def psnr(hr_img, sr_img):
    hr = np.clip(gpu_to_numpy(hr_img, is_squeeze=False), 0, 1) * 255
    sr = np.clip(gpu_to_numpy(sr_img, is_squeeze=False), 0, 1) * 255
    mse = ((hr - sr) ** 2).sum() / hr_img.numel()
    return 10 * np.log10(65025 / mse)
    

def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        Rotate([0, 90, 180, 270]),
        ToTensor()
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def valid_hr_transform(shape, upscale_factor):
    return Compose([
        ToTensor(),
        Resize((shape[0] // upscale_factor, shape[1] // upscale_factor), interpolation=Image.BICUBIC)
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)        
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

