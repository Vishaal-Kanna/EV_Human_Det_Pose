#!/usr/bin/evn python

import torch
import torchvision
from PIL import Image, ImageOps
import glob
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import zarr
import os
import torch.autograd as auto
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


def load_heatmap(imgfilename, device="cpu"):
    title, ext = os.path.splitext(os.path.basename(imgfilename))
    heatmap_filename = 'PoseTrack/training_data_DHP19/labels/'+ title + '.zarr'
    heatmap = zarr.load(heatmap_filename)
    return torch.from_numpy(1*np.transpose(heatmap, axes=(2, 0, 1))).float()

# def load_1D_heatmap(imgfilename, device="cpu"):
#     title, ext = os.path.splitext(os.path.basename(imgfilename))
#     heatmap_filename = 'PoseTrack/training_data_DHP19/labels/'+ title + '.zarr'
#     heatmap = zarr.load(heatmap_filename)
#     heatmap_1D = np.zeros()
#     for i in range(0,13):
#         x1, y1 = np.where(heatmap[:,:,i] == heatmap[:,:,i].max())
#     heatmap_1D
#     return torch.from_numpy(1*np.transpose(heatmap, axes=(2, 0, 1))).float()

# def load_1D_heatmap_batch(data, idxs, device="cpu"):
#     heatmap = [load_1D_heatmap(data[idx], device=device) for idx in idxs]
#     heatmap = torch.stack(heatmap, 0)
#     return heatmap.to(device)

def load_heatmap_batch(data, idxs, device="cpu"):
    heatmap = [load_heatmap(data[idx], device=device) for idx in idxs]
    heatmap = torch.stack(heatmap, 0)
    return heatmap.to(device)

def load_img(imgfilename, device="cpu"):
    img = Image.open(imgfilename)
    img = ImageOps.grayscale(img)
    transforms = [torchvision.transforms.ToTensor()]
    transformation = torchvision.transforms.Compose(transforms)
    img = transformation(img).to(device)
    return img

def load_img_batch(data, idxs, device="cpu"):
    img = [load_img(data[idx], device=device) for idx in idxs]
    img = torch.stack(img, 0)
    return img.to(device)

def load_img_for_test(data, idx, device="cpu"):
    img = load_img(data[idx], device=device)
    img = img.reshape((1,1,240,320))
    return img.to(device)

def load_gt_heatmap_for_test(data, idx, device="cpu"):
    gt_heatmap = load_heatmap(data[idx], device=device)
    return gt_heatmap.to(device)

def show_img(x):
    x = x.reshape(1,240,320)
    transform = torchvision.transforms.ToPILImage()
    return transform(x.float())

def show_heatmap(x,img):
    x = x.cpu().detach().numpy()
    x = x.reshape(13,240,320)
    x = np.sum(x,axis=0)
    plt.imshow(img)
    plt.imshow(x, alpha=.5)
    plt.show()

def error(test_output,test_gt_heatmap):
    test_output = test_output.cpu().detach().numpy()
    test_gt_heatmap = test_gt_heatmap.cpu().detach().numpy()
    test_output = test_output.reshape((13,240,320))
    e=0
    for i in range(0,test_output.shape[0]):
        if test_gt_heatmap[i].max()!=0:
            x2,y2 = np.where(test_gt_heatmap[i]==test_gt_heatmap[i].max())
            if test_output[i].max() != 0:
                x1, y1 = np.where(test_output[i] == test_output[i].max())
                e += (((x2 - x1)) ** 2 + ((y2 - y1)) ** 2) ** 0.5
            else:
                # x1, y1 = [test_output[i].shape[1]/2],[test_output[i].shape[0]/2]
                e += 2**0.5
        elif test_gt_heatmap[i].max()==0 and test_output[i].max()!=0:
            e += 2 ** 0.5

    e /= test_output.shape[0]
    return e

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.conv1 = torch.nn.Conv2d(1, 16, 3, 1, 1, dilation=1, bias=False)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 1, 1, dilation=1, bias=False)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, 1, 1, dilation=1, bias=False)
        self.conv4 = torch.nn.Conv2d(32, 32, 3, 1, 1, dilation=1, bias=False)
        self.conv5 = torch.nn.Conv2d(32, 64, 3, 1, 2, dilation=2, bias=False)
        self.conv6 = torch.nn.Conv2d(64, 64, 3, 1, 2, dilation=2, bias=False)
        self.conv7 = torch.nn.Conv2d(64, 64, 3, 1, 2, dilation=2, bias=False)
        self.conv8 = torch.nn.Conv2d(64, 64, 3, 1, 2, dilation=2, bias=False)
        self.conv9 = torch.nn.ConvTranspose2d(64, 32, 3, 2, 1, dilation=1, bias=False)
        self.conv10 = torch.nn.Conv2d(32, 32, 3, 1, 2, dilation=2, bias=False)
        self.conv11 = torch.nn.Conv2d(32, 32, 3, 1, 2, dilation=2, bias=False)
        self.conv12 = torch.nn.Conv2d(32, 32, 3, 1, 2, dilation=2, bias=False)
        self.conv13 = torch.nn.Conv2d(32, 32, 3, 1, 2, dilation=2, bias=False)
        self.conv14 = torch.nn.ConvTranspose2d(32, 16, 3, 2, 1, dilation=1, bias=False)
        self.conv15 = torch.nn.Conv2d(16, 16, 3, 1, 1, dilation=1, bias=False)
        self.conv16 = torch.nn.Conv2d(16, 16, 3, 1, 1, dilation=1, bias=False)
        self.conv17 = torch.nn.Conv2d(16, 13, 3, 1, 1, dilation=1, bias=False)

    def forward(self, x):

        input_size = x.size()
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        conv3_size = x.size()
        x = self.relu(self.pool(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x, output_size = conv3_size))
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
        x = self.relu(self.conv14(x, output_size=input_size))
        x = self.relu(self.conv15(x))
        x = self.relu(self.conv16(x))
        x = self.relu(self.conv17(x))

        return x

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=False):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class KLDiscretLoss(nn.Module):
    """
    "https://github.com/leeyegy/SimDR"
    """
    def __init__(self):
        super(KLDiscretLoss, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=1)  # [B,LOGITS]
        self.criterion_ = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        loss = torch.mean(self.criterion_(scores, labels), dim=1)
        return loss

    def forward(self, output_x, output_y, target_x, target_y):
        num_joints = output_x.size(1)
        loss = 0
        for idx in range(num_joints):
            coord_x_pred = output_x[:, idx]
            coord_y_pred = output_y[:, idx]
            coord_x_gt = target_x[:, idx]
            coord_y_gt = target_y[:, idx]
            # weight = target_weight[:, idx]
            loss += (self.criterion(coord_x_pred, coord_x_gt).mean())
            loss += (self.criterion(coord_y_pred, coord_y_gt).mean())

        return loss / num_joints

def load_checkpoint(model, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    checkpoint = torch.load(filename)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'],strict=False)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss = checkpoint['loss']
    print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))

    return model, optimizer, start_epoch, loss

def main():
    torch.cuda.empty_cache()
    writer = SummaryWriter(log_dir='log/DHP19_CNN')
    imgtrainfile = 'PoseTrack/training_data_DHP19/train.txt'
    with open(imgtrainfile, 'r') as file:
        data = file.read().splitlines()

    batch_size = 16
    train_idxs = np.arange(0, len(data), 1, dtype=int)

    network = model()

    device = torch.device("cuda")
    network = network.to(device)

    JointsLoss = JointsMSELoss().to(device)
    optimizer = torch.optim.RMSprop(network.parameters(), lr=1e-3)

    network, optimizer, e, loss = load_checkpoint(network, filename='PoseTrack/training_data_DHP19/Weights_DHP19/weights_20.tar')

    max_epoch = 20

    # e = 1
    #
    # while e <= max_epoch:
    #     if e>=11 and e<=15:
    #         optimizer = torch.optim.RMSprop(network.parameters(), lr=1e-4)
    #     elif e>=16 and e<=20:
    #         optimizer = torch.optim.RMSprop(network.parameters(), lr=1e-5)
    #     print("Epoch: ", e)
    #     np.random.shuffle(train_idxs)
    #     range_ = tqdm(np.array_split(train_idxs, len(train_idxs) / (batch_size-1)))
    #     loss_sum=0
    #     for i, idxs in enumerate(range_):
    #         batch_img = load_img_batch(data, idxs, device)
    #         batch_heatmap = load_heatmap_batch(data, idxs, device)
    #         batch_predictions = network(batch_img)
    #         loss = JointsLoss(batch_predictions, batch_heatmap)
    #         optimizer.zero_grad()
    #         loss_sum+= loss.item()
    #         loss_avg = loss_sum/(i+1)
    #         range_.set_postfix(loss=loss_avg)
    #         loss.backward()
    #         optimizer.step()
    #         writer.add_scalar('Loss/train', loss.item(), (e-1)*31481+i)
    #     state = {'epoch': e + 1, 'state_dict': network.state_dict(),'optimizer': optimizer.state_dict(),'loss': loss, }
    #     torch.save(state, 'PoseTrack/training_data_DHP19/Weights_DHP19/weights_{}.tar'.format(e))
    #     e += 1

    test_error = 0
    for k in range(0,len(data),4):
        test_image = load_img_for_test(data, k, device)
        test_gt_heatmap = load_gt_heatmap_for_test(data, k, device)
        test_output = network(test_image)
        test_error+=error(test_output,test_gt_heatmap)
        # im = show_img(test_image)
        # show_heatmap(test_gt_heatmap,im)
        # show_heatmap(test_output,im)
    test_error/=(len(data)/4)
    print(test_error)

if __name__ == '__main__':
    main()
