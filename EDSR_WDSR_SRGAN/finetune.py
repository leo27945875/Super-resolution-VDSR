import os
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim.lr_scheduler as schedule
import torch_optimizer as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from warmup_scheduler import GradualWarmupScheduler
import ttach as tta

from model import *
from loss import *
from data_utils import *


if __name__ == "__main__":
    UPSCALE_FACTOR  = 3
    LEARNING_RATE_G = 1e-4
    LEARNING_RATE_D = 1e-4
    CROP_SIZE       = 96
    UPSCALE_FACTOR  = 3
    NUM_EPOCHS      = 1000
    BATCH_SIZE      = 20
    IMAGE_DIR       = "data/origin_hr_images"
    MODEL_TYPE      = "SRResnet"
    MODEL_NAME      = "SRResnet_x3_epoch=205_PSNR=25.0052.pth"
    DO_MIXUP        = 1
    NUM_DISCRIMINATOR_TRAIN = 2
    
    train_set    = TrainDatasetFromFolder(IMAGE_DIR, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    n_iter = (len(train_set) // BATCH_SIZE + 1) * NUM_EPOCHS
    
    gen = eval(f"{MODEL_TYPE}({UPSCALE_FACTOR})")
    dis = Discriminator()
    
    if MODEL_NAME:
        gen.load_state_dict(torch.load(os.path.join("epochs", MODEL_NAME)))
        
    criterionG = GeneratorLoss()
    criterionD = DiscriminatorLoss()
    
    optimizerG = optim.RAdam(gen.parameters(), lr=LEARNING_RATE_G, betas=(0, 0.9))
    optimizerD = optim.RAdam(dis.parameters(), lr=LEARNING_RATE_D, betas=(0, 0.9))
    
    schedulerG = schedule.StepLR(optimizerG, int(n_iter * 0.3), gamma=0.5)
    schedulerD = schedule.StepLR(optimizerD, int(n_iter * 0.3), gamma=0.5)
    
    schedulerG = GradualWarmupScheduler(optimizerG, 1, n_iter // 20, after_scheduler=schedulerG)
    schedulerD = GradualWarmupScheduler(optimizerD, 1, n_iter // 20, after_scheduler=schedulerD)
    
    # plot_scheduler(schedulerG, n_iter, "Gen LR Scheduler")
    # plot_scheduler(schedulerD, n_iter * NUM_DISCRIMINATOR_TRAIN, "Dis LR Scheduler")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gen.to(device)
    dis.to(device)
    criterionG.to(device)
    criterionD.to(device)
    
    tta_transform = tta.Compose([
        tta.HorizontalFlip(),
        tta.VerticalFlip()
    ])
    
    tta_gen = tta.SegmentationTTAWrapper(gen, tta_transform)
    tta_dis = tta.ClassificationTTAWrapper(dis, tta_transform)
    history = []
    img_test  = plt.imread(r'data\testing_lr_images\09.png')
    img_valid = plt.imread(r'data\valid_hr_images\t20.png')
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'lossG': 0, 'lossD': 0, 
                           'gradeHR': 0, 'gradeSR': 0, 'psnr': 0}
        
        # Train a epoch:
        gen.train()
        dis.train()
        for lr_img, hr_img in train_bar:
            running_results['batch_sizes'] += BATCH_SIZE
            train_iter = iter(train_loader)
            for i in range(NUM_DISCRIMINATOR_TRAIN):
                gen.requires_grad_(False)
                dis.requires_grad_(True)
                optimizerD.zero_grad()
                
                dis_lr_img, dis_hr_img = train_iter.next()
                if DO_MIXUP:
                    dis_lr_img, dis_hr_img = MixUp()(dis_lr_img, dis_hr_img, 
                                                     dis_lr_img, dis_hr_img)
                    
                dis_lr_img = dis_lr_img.type(torch.FloatTensor).to(device)
                dis_hr_img = dis_hr_img.type(torch.FloatTensor).to(device)
                dis_sr_img = tta_gen(dis_lr_img).detach()
                
                sr_label = dis(dis_sr_img)
                hr_label = tta_dis(dis_hr_img)
                
                lossD = criterionD(dis, sr_label, hr_label, dis_sr_img, dis_hr_img)
                lossD.backward()
                optimizerD.step()
                schedulerD.step()
            
            gen.requires_grad_(True)
            dis.requires_grad_(False)
            optimizerG.zero_grad()
            
            if DO_MIXUP:
                lr_img, hr_img = MixUp()(lr_img, hr_img, 
                                         lr_img, hr_img)
                
            lr_img = lr_img.type(torch.FloatTensor).to(device)
            hr_img = hr_img.type(torch.FloatTensor).to(device)
            sr_img = tta_gen(lr_img)
            
            sr_label = dis(sr_img)
            lossG = criterionG(sr_label, sr_img, hr_img)
            lossG.backward()
            optimizerG.step()
            schedulerG.step()
            
            running_results['lossD'  ] += lossD.item() * BATCH_SIZE
            running_results['lossG'  ] += lossG.item() * BATCH_SIZE
            running_results['gradeHR'] += torch.sigmoid(hr_label.mean()).item() * BATCH_SIZE
            running_results['gradeSR'] += torch.sigmoid(sr_label.mean()).item() * BATCH_SIZE
            running_results['psnr'   ] += psnr(hr_img, sr_img) * BATCH_SIZE
            train_bar.set_description(desc='[%d/%d] LossG: %.4f, LossD: %.4f, GradeHR: %.4f, GradeSR: %.4f, PSNR: %.4f' % (
                epoch, NUM_EPOCHS,
                running_results['lossG'  ] / running_results['batch_sizes'],
                running_results['lossD'  ] / running_results['batch_sizes'],
                running_results['gradeHR'] / running_results['batch_sizes'],
                running_results['gradeSR'] / running_results['batch_sizes'],
                running_results['psnr'   ] / running_results['batch_sizes']
            ))
    
        # Save model parameters:
        psnr_now = running_results['psnr'] / running_results['batch_sizes']
        filenameG = 'epochs/netG_x%d_epoch=%d_PSNR=%.4f.pth' % (UPSCALE_FACTOR, epoch, psnr_now)
        filenameD = 'epochs/netD_x%d_epoch=%d_PSNR=%.4f.pth' % (UPSCALE_FACTOR, epoch, psnr_now)
        torch.save(gen.state_dict(), filenameG)
        torch.save(dis.state_dict(), filenameD)
        history.append(running_results)
        
        # Test model:
        if epoch % 5 == 0:
            with torch.no_grad():
                gen.eval()
                
                # Plot up-scaled testing image:
                test_lr_img = torch.from_numpy(img_test.transpose(2, 0, 1)).unsqueeze(0).to(device)
                test_sr_img = gen(test_lr_img)
                plot_hr_lr(test_sr_img, test_lr_img)
                
                # Compute PSNR of validation image:
                valid_hr_img = ToTensor()(img_valid)
                valid_lr_img = valid_hr_transform(img_valid.shape, UPSCALE_FACTOR)(img_valid).to(device)
                valid_sr_img = gen(valid_lr_img)
                psnr_valid = psnr(valid_hr_img, valid_sr_img)
                
                # Print PSNR:
                print('\n' + '-' * 50)
                print(f"PSNR of Validation = {psnr_valid}")
                print('-' * 50 + '\n')
    
    



















