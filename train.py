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


def load_heatmap(imgfilename, device="cpu"):
    title, ext = os.path.splitext(os.path.basename(imgfilename))
    heatmap_filename = 'PoseTrack/training_data/labels_bak/'+ title + '.zarr'
    heatmap = zarr.load(heatmap_filename)
    # print(heatmap.shape)
    # quit()
    return torch.from_numpy(np.transpose(heatmap, axes=(2, 0, 1))).float()

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
    img = img.reshape((1,1,344,180))
    return img.to(device)

def load_gt_heatmap_for_test(data, idx, device="cpu"):
    gt_heatmap = load_heatmap(data[idx], device=device)
    return gt_heatmap.to(device)

def show_img(x):
    x = x.reshape(1,344,180)
    transform = torchvision.transforms.ToPILImage()
    return transform(x.float())

def show_heatmap(x,img):
    x = x.cpu().detach().numpy()
    x = x.reshape(17,344,180)
    x = np.sum(x,axis=0)
    plt.imshow(img)
    plt.imshow(x, alpha=.5)
    plt.show()

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
        self.conv17 = torch.nn.Conv2d(16, 17, 3, 1, 1, dilation=1, bias=False)

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

class model_bak(torch.nn.Module):
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
        self.conv9 = torch.nn.Conv2d(64, 32, 3, 1, 1, dilation=1, bias=False)
        self.conv10 = torch.nn.Conv2d(32, 32, 3, 1, 2, dilation=2, bias=False)
        self.conv11 = torch.nn.Conv2d(32, 32, 3, 1, 2, dilation=2, bias=False)
        self.conv12 = torch.nn.Conv2d(32, 32, 3, 1, 2, dilation=2, bias=False)
        self.conv13 = torch.nn.Conv2d(32, 32, 3, 1, 2, dilation=2, bias=False)
        self.conv14 = torch.nn.Conv2d(32, 16, 3, 1, 1, dilation=1, bias=False)
        self.conv15 = torch.nn.Conv2d(16, 16, 3, 1, 1, dilation=1, bias=False)
        self.conv16 = torch.nn.Conv2d(16, 16, 3, 1, 1, dilation=1, bias=False)
        self.conv17 = torch.nn.Conv2d(16, 17, 3, 1, 1, dilation=1, bias=False)

    def forward(self, x):

        input_size = x.size()
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # conv3_size = x.size()
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
        x = self.relu(self.conv14(x))
        x = self.relu(self.conv15(x))
        x = self.relu(self.conv16(x))
        x = self.relu(self.conv17(x))

        return x

def get_gradients(model, optimizer, data, target):
    model.eval()
    criterion = nn.MSELoss()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward(retain_graph=True)
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(name)
            print(param.data.cpu().numpy().shape)
            print('gradient is \t', param.grad, '\trequires grad: ', param.requires_grad)
            print(' ')

    optimizer.zero_grad()
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(name)
            grad = auto.grad(loss, param, retain_graph=True, only_inputs=False)
            print('another gradit\t', grad)
            print(' ')

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

def main():
    torch.cuda.empty_cache()
    imgtrainfile = 'PoseTrack/training_data/train_bak.txt'
    with open(imgtrainfile, 'r') as file:
        data = file.read().splitlines()

    batch_size = 16
    train_idxs = np.arange(0, len(data), 1, dtype=int)

    network = model_bak()
    # print(network)
    # quit()

    device = torch.device("cuda")
    network = network.to(device)

    mse = nn.MSELoss()
    JointsLoss = JointsMSELoss().to(device)
    optimizer = torch.optim.RMSprop(network.parameters(), lr=1e-4)

    # checkpoint = torch.load('PoseTrack/training_data/Weights/weights_31.tar')
    # network.load_state_dict(checkpoint['state_dict'])
    # e = checkpoint['epoch']
    # loss = checkpoint['loss']

    max_epoch = 50

    e = 1

    while e <= max_epoch:
        print("Epoch: ", e)
        np.random.shuffle(train_idxs)
        range_ = tqdm(np.array_split(train_idxs, len(train_idxs) / batch_size))
        # loss_sum=0
        for i, idxs in enumerate(range_):
            batch_img = load_img_batch(data, idxs, device)
            batch_heatmap = load_heatmap_batch(data, idxs, device)
            batch_predictions = network(batch_img)
            loss = JointsLoss(batch_predictions, batch_heatmap)

            # get_gradients(network, optimizer, batch_img, batch_heatmap)
            optimizer.zero_grad()
            # loss.register_hook(lambda grad: print(grad))
            # loss_sum+= loss.item()
            # loss_avg = loss_sum/(i+1)
            range_.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()
        if (e-1)%3==0:
            state = {'epoch': e + 1, 'state_dict': network.state_dict(),'optimizer': optimizer.state_dict(),'loss': loss, }
            torch.save(state, 'PoseTrack/training_data/Weights/weights_{}.tar'.format(e))
        e += 1

    test_image = load_img_for_test(data, 150, device)
    test_gt_heatmap = load_gt_heatmap_for_test(data, 150, device)
    # test_output = network(test_image)
    im = show_img(test_image)
    show_heatmap(test_gt_heatmap,im)
    # show_heatmap(test_output)

if __name__ == '__main__':
    main()
