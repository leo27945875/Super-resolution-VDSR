import torch
import torch.nn.functional as F
from torch import nn
from torch import autograd
from torchvision.models.vgg import vgg19
from torchvision.transforms import Normalize


class TotalLoss(nn.Module):
    def __init__(self, tv_loss_rate=1e-5, perception_rate=0):
        super(TotalLoss, self).__init__()
        self.t_rate = tv_loss_rate
        self.p_rate = perception_rate
        self.l1_loss = nn.L1Loss()
        self.tv_loss = TVLoss()
        self.loss_network = None
        
        if perception_rate:
            print("Use perception")
            vgg = vgg19(pretrained=True)
            loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
            for param in loss_network.parameters():
                param.requires_grad = False
                
            self.loss_network = loss_network
    
    def forward(self, fake_imgs, real_imgs):
        l1_loss = self.l1_loss(fake_imgs, real_imgs)
        tv_loss = self.tv_loss(fake_imgs)
        
        if self.loss_network:
            perception_loss = self.perception(fake_imgs, real_imgs)
            return l1_loss + self.p_rate * perception_loss + self.t_rate * tv_loss
        else:
            return l1_loss + self.t_rate * tv_loss
    
    def perception(self, fake_imgs, real_imgs):
        normalizer = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        fake_imgs = normalizer(fake_imgs)
        real_imgs = normalizer(real_imgs)
        return F.l1_loss(self.loss_network(fake_imgs), self.loss_network(real_imgs))


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
            
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()
        self.tv_loss = TVLoss()

    def forward(self, fake_labels, fake_imgs, real_imgs):
        adversarial_loss = - fake_labels.mean()
        perception_loss  = self.perception(fake_imgs, real_imgs)
        # image_loss       = self.l1_loss(fake_imgs, real_imgs)
        tv_loss          = self.tv_loss(fake_imgs)
        return perception_loss + 5e-3 * adversarial_loss + 1e-7 * tv_loss
    
    def perception(self, fake_imgs, real_imgs):
        normalizer = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        fake_imgs = normalizer(fake_imgs)
        real_imgs = normalizer(real_imgs)
        return F.l1_loss(self.loss_network(fake_imgs), self.loss_network(real_imgs))
    

class DiscriminatorLoss(nn.Module):
    def __init__(self, lam=10.):
        super(DiscriminatorLoss, self).__init__()
        self.lam = lam
    
    def __call__(self, model, fake_labels, real_labels, fake_imgs, real_imgs):
        gp = self.gradient_panelty(model, fake_imgs.detach(), real_imgs.detach())
        return fake_labels.mean() - real_labels.mean() + self.lam * gp

    def gradient_panelty(self, model, fake_imgs, real_imgs):
        batch_size, c, h, w = fake_imgs.shape
        t = torch.rand((batch_size, 1, 1, 1)).repeat(1, c, h, w).to(fake_imgs.device)
        mids = (real_imgs * t + fake_imgs * (1 - t)).requires_grad_()
        outs = model(mids)
        grad = autograd.grad(outputs=outs, inputs=mids, 
                             grad_outputs=torch.ones_like(outs),
                             retain_graph=True, create_graph=True,
                             only_inputs=True)[0]
        grad = grad.view(grad.shape[0], -1)
        gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return gp











