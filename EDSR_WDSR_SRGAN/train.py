import argparse
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim.lr_scheduler as schedule
import torch_optimizer as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from warmup_scheduler import GradualWarmupScheduler
import ttach as tta

from data_utils import *
from model import *
from loss import *


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--model_type'    , default='SRResnet', type=str  , help='WDSR|EDSR')
parser.add_argument('--learning_rate' , default=1e-4  , type=float, help='learning rate')
parser.add_argument('--crop_size'     , default=96    , type=int  , help='training images crop size')
parser.add_argument('--upscale_factor', default=3     , type=int  , help='super resolution upscale factor', choices=[2, 3])
parser.add_argument('--num_epochs'    , default=1000  , type=int  , help='train epoch number')
parser.add_argument('--batch_size'    , default=64    , type=int  , help='batch size for training process')
parser.add_argument('--tv_loss_rate'  , default=1e-5  , type=float, help='batch size for training process')
parser.add_argument('--last_epoch'    , default=-1    , type=int  , help='last epoch')
parser.add_argument('--model_name'    , default=''    , type=str  , help='generator model epoch name')
parser.add_argument('--train_img_dir' , default='data/origin_hr_images', type=str, help='folder of training images')


def main(MODEL_TYPE, LEARNING_RATE, CROP_SIZE, UPSCALE_FACTOR, NUM_EPOCHS, BATCH_SIZE, IMAGE_DIR, LAST_EPOCH, MODEL_NAME='', TV_LOSS_RATE=1e-3):
    
    global net, history, lr_img, hr_img, test_lr_img, test_sr_img, valid_hr_img, valid_sr_img, valid_lr_img, scheduler, optimizer
    
    train_set    = TrainDatasetFromFolder(IMAGE_DIR, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=BATCH_SIZE, shuffle=True)
    n_iter = (len(train_set) // BATCH_SIZE + 1) * NUM_EPOCHS

    net = eval(f"{MODEL_TYPE}({UPSCALE_FACTOR})")
    criterion = TotalLoss(TV_LOSS_RATE)
    optimizer = optim.RAdam(net.parameters(), lr=LEARNING_RATE)
    scheduler = schedule.StepLR(optimizer, int(n_iter * 0.3), gamma=0.5, last_epoch=LAST_EPOCH)
    if LAST_EPOCH == -1:
        scheduler = GradualWarmupScheduler(optimizer, 1, n_iter // 50, after_scheduler=scheduler)
        
    # plot_scheduler(scheduler, n_iter)

    if MODEL_NAME:
        net.load_state_dict(torch.load('epochs/' + MODEL_NAME))
        print(f"# Loaded model: [{MODEL_NAME}]")
    
    print(f'# {MODEL_TYPE} parameters:', sum(param.numel() for param in net.parameters()))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    criterion.to(device)
    
    tta_transform = tta.Compose([
        tta.HorizontalFlip(),
        tta.VerticalFlip()
    ])
    
    # Train model:
    tta_net = tta.SegmentationTTAWrapper(net, tta_transform)
    history = []
    img_test  = plt.imread(r'data\testing_lr_images\09.png')
    img_valid = plt.imread(r'data\valid_hr_images\t20.png')
    test_lr_img = torch.from_numpy(img_test.transpose(2, 0, 1)).unsqueeze(0).to(device)
    valid_hr_img = ToTensor()(img_valid)
    valid_lr_img = valid_hr_transform(img_valid.shape, UPSCALE_FACTOR)(img_valid).to(device)
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'loss': 0, 'psnr': 0}
        
        # Train a epoch:
        net.train()
        for lr_img, hr_img in train_bar:
            running_results['batch_sizes'] += BATCH_SIZE
            optimizer.zero_grad()
            
            lr_img, hr_img = MixUp()(lr_img, hr_img, lr_img, hr_img)
            lr_img = lr_img.type(torch.FloatTensor).to(device)
            hr_img = hr_img.type(torch.FloatTensor).to(device)
            sr_img = tta_net(lr_img)
            
            loss = criterion(sr_img, hr_img)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_results['loss'] += loss.item() * BATCH_SIZE
            running_results['psnr'] += psnr(hr_img, sr_img) * BATCH_SIZE
            train_bar.set_description(desc='[%d/%d] Loss: %.4f, PSNR: %.4f' % (
                epoch, NUM_EPOCHS,
                running_results['loss'] / running_results['batch_sizes'],
                running_results['psnr'] / running_results['batch_sizes']
            ))
    
        # Save model parameters:
        psnr_now = running_results['psnr'] / running_results['batch_sizes']
        filename = f'epochs/{MODEL_TYPE}_x%d_epoch=%d_PSNR=%.4f.pth' % (UPSCALE_FACTOR, epoch, psnr_now)
        torch.save(net.state_dict(), filename)
        history.append(running_results)
        
        # Test model:
        if epoch % 5 == 0:
            with torch.no_grad():
                net.eval()
                
                # Plot up-scaled testing image:
                test_sr_img = net(test_lr_img)
                plot_hr_lr(test_sr_img, test_lr_img)
                
                # Compute PSNR of validation image:
                valid_sr_img = net(valid_lr_img)
                psnr_valid = psnr(valid_hr_img, valid_sr_img)
                
                # Print PSNR:
                print('\n' + '-' * 50)
                print(f"PSNR of Validation = {psnr_valid}")
                print('-' * 50 + '\n')
                
    torch.save(optimizer.state_dict(), f'optimizer_{MODEL_TYPE}_epoch={NUM_EPOCHS}.pth')
    torch.save(scheduler.state_dict(), f'scheduler_{MODEL_TYPE}_epoch={NUM_EPOCHS}.pth')


if __name__ == '__main__':
    opt = parser.parse_args()
    
    MODEL_TYPE     = opt.model_type
    LEARNING_RATE  = opt.learning_rate
    CROP_SIZE      = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS     = opt.num_epochs
    BATCH_SIZE     = opt.batch_size
    IMAGE_DIR      = opt.train_img_dir
    MODEL_NAME     = opt.model_name
    TV_LOSS_RATE   = opt.tv_loss_rate
    LAST_EPOCH     = opt.last_epoch
    
    main(MODEL_TYPE, 
         LEARNING_RATE, 
         CROP_SIZE, 
         UPSCALE_FACTOR, 
         NUM_EPOCHS, 
         BATCH_SIZE,
         IMAGE_DIR, 
         LAST_EPOCH,
         MODEL_NAME, 
         TV_LOSS_RATE)
            
        
        
        
        
        
        
        
 
