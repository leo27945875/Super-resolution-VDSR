import argparse
import time
import os
import glob

import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from model import EDSR, WDSR, SRResnet
from data_utils import plot_hr_lr, gpu_to_numpy


parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=3, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode' , default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_dir' , default=r'data\testing_lr_images', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='EDSR_x3_epoch=912_PSNR=28.5340.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE  = True if opt.test_mode == 'GPU' else False
IMAGE_DIR  = opt.image_dir
MODEL_NAME = opt.model_name


def Test(MODEL_NAME, UPSCALE_FACTOR, is_save=False, IMAGE_DIR=r'data\testing_lr_images', TEST_MODE=True):
    if type(MODEL_NAME) is EDSR or type(MODEL_NAME) is WDSR or type(MODEL_NAME) is SRResnet:
        model = MODEL_NAME

    else:
        model = EDSR(UPSCALE_FACTOR).eval()  
        if TEST_MODE:
            model.cuda()
            model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
        else:
            model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))
    
    print('\n----------------------------------------------------------')
    imgs = []
    with torch.no_grad():
        model.eval()
        for image_name in glob.glob(os.path.join(IMAGE_DIR, '*.*')):
            image = Image.open(image_name)
            image = ToTensor()(image).unsqueeze(0)
            if TEST_MODE:
                image = image.cuda()
            
            start = time.time()
            out = model(image)
            elapsed = (time.time() - start)
            
            out_img = ToPILImage()(torch.clip(out[0], 0, 1))
            if is_save:
                out_img.save(f'data/testing_sr_images/{os.path.basename(image_name)}')
            
            sr_img = gpu_to_numpy(out[0], is_squeeze=False)
            imgs.append(sr_img)
            plot_hr_lr(sr_img, image)
            print('cost time: ' + str(elapsed) + 's')
            
    return imgs


if __name__ == '__main__':
    imgs = Test(MODEL_NAME, UPSCALE_FACTOR, True, IMAGE_DIR, TEST_MODE)