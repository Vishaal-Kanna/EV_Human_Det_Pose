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
from config import cfg
from config import update_config
import logging
import argparse
from torch.utils.tensorboard import SummaryWriter


def load_heatmap(imgfilename, device="cpu"):
    title, ext = os.path.splitext(os.path.basename(imgfilename))
    heatmap_filename = 'PoseTrack/training_data_DHP19/labels/'+ title + '.zarr'
    heatmap = zarr.load(heatmap_filename)
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
                e += (((x2 - x1) / test_output[i].shape[1]) ** 2 + ((y2 - y1) / test_output[i].shape[0]) ** 2) ** 0.5
            else:
                # x1, y1 = [test_output[i].shape[1]/2],[test_output[i].shape[0]/2]
                e += 2**0.5
        elif test_gt_heatmap[i].max()==0 and test_output[i].max()!=0:
            e += 2 ** 0.5

    e /= test_output.shape[0]
    return e

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

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        super(PoseResNet, self).__init__()

        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        self.coord_representation = cfg.MODEL.COORD_REPRESENTATION
        assert  self.coord_representation in ['heatmap', 'simdr', 'sa-simdr'], 'only heatmap or simdr or sa-simdr supported ~ '

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv8 = torch.nn.Conv2d(17, 16, 3, 1, (0,1), dilation=1, bias=False)
        self.conv9 = torch.nn.Conv2d(16, 16, 3, 1, (0,1), dilation=1, bias=False)
        self.upsample1 = torch.nn.ConvTranspose2d(16, 16, 3, 2, 1, dilation=1, bias=False)
        self.conv10 = torch.nn.Conv2d(16, 16, 3, 1, 2, dilation=2, bias=False)
        self.conv11 = torch.nn.Conv2d(16, 16, 3, 1, 2, dilation=2, bias=False)
        self.upsample2 = torch.nn.ConvTranspose2d(16, 16, 3, 2, 1, dilation=1, bias=False)
        self.conv12 = torch.nn.Conv2d(16, 16, 3, 1, 1, dilation=1, bias=False)
        self.conv13 = torch.nn.Conv2d(16, 13, 3, 1, 1, dilation=1, bias=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        # head
        if self.coord_representation == 'simdr' or  self.coord_representation == 'sa-simdr':
            self.mlp_head_x = nn.Linear(cfg.MODEL.HEAD_INPUT, int(cfg.MODEL.IMAGE_SIZE[0]*cfg.MODEL.SIMDR_SPLIT_RATIO))
            self.mlp_head_y = nn.Linear(cfg.MODEL.HEAD_INPUT, int(cfg.MODEL.IMAGE_SIZE[1]*cfg.MODEL.SIMDR_SPLIT_RATIO))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)

        x = self.final_layer(x)

        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.relu(self.upsample1(x, output_size = (120,160)))
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x))
        x = self.relu(self.upsample2(x, output_size=(240,320)))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))

        if self.coord_representation == 'simdr' or self.coord_representation == 'sa-simdr':
            x = rearrange(x, 'b c h w -> b c (h w)')
            pred_x = self.mlp_head_x(x)
            pred_y = self.mlp_head_y(x)
            return pred_x, pred_y
        elif self.coord_representation == 'heatmap':
            return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
    return model

def main():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default = 'PoseTrack/training_data_DHP19/config_files/pose_resnet.yaml',
                        required=False,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    update_config(cfg, args)


    # input = torch.rand(1,1,240,320)
    # network = get_pose_net(cfg, False)
    # op = network(input)
    # print(op.shape)
    # quit()
    writer = SummaryWriter(log_dir='log/pose_resnet')

    torch.cuda.empty_cache()
    imgtrainfile = 'PoseTrack/training_data_DHP19/train.txt'
    with open(imgtrainfile, 'r') as file:
        data = file.read().splitlines()

    batch_size = 4
    train_idxs = np.arange(0, len(data), 1, dtype=int)

    network = get_pose_net(cfg, False)

    device = torch.device("cuda")
    network = network.to(device)

    JointsLoss = JointsMSELoss().to(device)

    network, optimizer, e, loss = load_checkpoint(network, filename='PoseTrack/training_data_DHP19/Weights/weights_3.tar')
    optimizer = torch.optim.RMSprop(network.parameters(), lr=1e-3)

    max_epoch = 15

    # e = 1
    #
    # while e <= max_epoch:
    #     print("Epoch: ", e)
    #     np.random.shuffle(train_idxs)
    #     range_ = tqdm(np.array_split(train_idxs, len(train_idxs) / batch_size))
    #     loss_sum=0
    #     for i, idxs in enumerate(range_):
    #         batch_img = load_img_batch(data, idxs, device)
    #         batch_heatmap = load_heatmap_batch(data, idxs, device)
    #         batch_predictions = network(batch_img)
    #         loss = JointsLoss(batch_predictions, batch_heatmap)
    #
    #         # get_gradients(network, optimizer, batch_img, batch_heatmap)
    #         optimizer.zero_grad()
    #         # loss.register_hook(lambda grad: print(grad))
    #         loss_sum+= loss.item()
    #         loss_avg = loss_sum/(i+1)
    #         range_.set_postfix(loss=loss_avg)
    #         loss.backward()
    #         optimizer.step()
    #         writer.add_scalar('Loss/train', loss.item(), (e-1)*31481+i)
    #     state = {'epoch': e + 1, 'state_dict': network.state_dict(),'optimizer': optimizer.state_dict(),'loss': loss, }
    #     torch.save(state, 'PoseTrack/training_data_DHP19/Weights_resnet/weights_{}.tar'.format(e))
    #     e += 1

    test_error = 0
    for k in range(0,len(data),2):
        test_image = load_img_for_test(data, k, device)
        test_gt_heatmap = load_gt_heatmap_for_test(data, k, device)
        test_output = network(test_image)
        test_error+=error(test_output,test_gt_heatmap)
        # im = show_img(test_image)
        # show_heatmap(test_gt_heatmap,im)
        # show_heatmap(test_output,im)
    test_error/=(len(data)/2)
    print(test_error)

if __name__ == '__main__':
    main()
