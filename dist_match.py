import torch
import torch.nn as nn
import torchvision.utils as vutils
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

import os
import math
import random
import argparse

class ConvNet(nn.Module):
    def __init__(self, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling'):
        super(ConvNet, self).__init__()

        channel = 3
        im_size = (32,32)

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]

    def forward(self, x):
        # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2


        return nn.Sequential(*layers), shape_feat

class Trainer():
    def __init__(self, params, train_loader):
        self.p = params

        self.train_loader = train_loader
        self.gen = self.inf_train_gen()

        self.ims = torch.randn(10*self.p.num_ims,3,32,32).to(self.p.device)

        if self.p.init_ims:
            self.init_ims()

        self.ims = torch.nn.Parameter(self.ims)
        self.labels = torch.arange(10, device=self.p.device).repeat(self.p.num_ims,1).T.flatten()
        self.opt_ims = torch.optim.Adam([self.ims], lr=self.p.lr)

        self.models = []
        for _ in range(self.p.num_models):
        	m = ConvNet()
        	self.models.append(m)
        
        ### Make Log Dirs
        if not os.path.isdir(self.p.log_dir):
            os.mkdir(self.p.log_dir)

        path = os.path.join(self.p.log_dir, 'images')
        if not os.path.isdir(path):
            os.mkdir(path)

        path = os.path.join(self.p.log_dir, 'checkpoints')
        if not os.path.isdir(path):
            os.mkdir(path)

    def inf_train_gen(self):
        while True:
            for data in self.train_loader:
                yield data

    def init_ims(self):
        for c in range(10):
            X = torch.load(os.path.join('../data/', f'data_class_{c}.pt'))
            perm = torch.randperm(X.shape[0])[:self.p.num_ims]
            xc = X[perm]
            self.ims[c*self.p.num_ims:(c+1)*self.p.num_ims] = xc


    def log_interpolation(self, step):
        path = os.path.join(self.p.log_dir, 'images/synth')
        if not os.path.isdir(path):
            os.mkdir(path)
        ims = torch.tanh(self.ims)
        torchvision.utils.save_image(
            vutils.make_grid(ims, nrow=self.p.num_ims, padding=2, normalize=True)
            , os.path.join(path, f'{step}.png'))


    def save(self):
        path = os.path.join(self.p.log_dir, 'checkpoints')
        if not os.path.isdir(path):
            os.mkdir(path)
        file_name = os.path.join(path, 'data.pt')
        ims = torch.tanh(self.ims)
        torch.save(ims.cpu(), file_name)

        file_name = os.path.join(path, 'labels.pt')
        torch.save(self.labels.cpu(), file_name)

    def load_ims(self):
        path = os.path.join(self.p.log_dir, 'checkpoints', 'data.pt')
        if os.path.exists(path):
            self.ims = torch.load(path)
            self.ims = torch.nn.Parameter(self.ims)
        return os.path.exists(path)

    
    def total_variation_loss(self, img, weight=1, four=True):
        bs_img, c_img, h_img, w_img = img.size()

        tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()

        tv = weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

        if four:
            tv_h4 = torch.pow(img[:,:,:-1,:]-img[:,:,1:,:], 2).sum()
            tv_w4 = torch.pow(img[:,:,:,:-1]-img[:,:,:,1:], 2).sum()
            tv = tv + weight*(tv_h4+tv_w4)/(bs_img*c_img*h_img*w_img)
            tv = tv/2

        return tv

    def sample_model(self):
    	m = self.models[random.randint(0,self.p.num_models-1)]
    	return m.eval().to(self.p.device)

    def train_ims_cw(self):
        print('############## Training Images ##############',flush=True)
        self.ims.requires_grad = True

        for t in range(self.p.niter):
            loss = torch.tensor(0.0).to(self.p.device)
            data, labels = next(self.gen)
            for c in range(10):
                d_c = data[labels == c].to(self.p.device)
                ims = self.ims[c*self.p.num_ims:(c+1)*self.p.num_ims]
                model = self.sample_model()

                encX = model(d_c)
                encY = model(ims)

                mmd = torch.norm(encX.mean(dim=0)-encY.mean(dim=0))

                if self.p.corr:
                    corr = self.total_variation_loss(torch.tanh(ims))
                else:
                    corr = torch.zeros(1)

                loss = loss + mmd

                if self.p.corr:
                    loss = loss + self.p.corr_coef*corr

            self.opt_ims.zero_grad()
            loss.backward()
            self.opt_ims.step()
        
            if (t%100) == 0:
                s = '[{}|{}] Loss: {:.4f}, MMD: {:.4f}'.format(t, self.p.niter, loss.item(), mmd.item())
                if self.p.corr:
                    s += ', Corr: {:.4f}'.format(corr.item())
                print(s,flush=True)
                self.log_interpolation(t)

        self.save()
        self.ims.requires_grad = False


    def train(self):
        self.ims.requires_grad = False
        self.train_ims_cw()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Simple Dist Match')
    # General Training
    parser.add_argument('--batch-size', type=int, default= 256)
    parser.add_argument('--niter', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_ims', type=int, default=10)
    parser.add_argument('--num_models', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--corr', type=bool, default=False)
    parser.add_argument('--corr_coef', type=float, default=1)
    parser.add_argument('--init_ims', type=bool, default=False)

    args = parser.parse_args()

    train_kwargs = {'batch_size': args.batch_size, 'shuffle':True}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), 
            (0.5, 0.5, 0.5))
        ])
    dataset1 = datasets.CIFAR10('../data/', train=True, download=True,
                        transform=transform)
    

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)

    trainer = Trainer(args, train_loader)
    trainer.train()
    

if __name__ == '__main__':
    main()