import torch
import torch.nn as nn
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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

class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
            else:
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))

        else:
            self.mean = mean.cuda()
        if std is None:
            if torch.cuda.is_available():
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
            else:
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std.cuda()

    def forward(self, boxes, deltas):

        widths  = boxes[:, :, 2] - boxes[:, :, 0].cuda()
        heights = boxes[:, :, 3] - boxes[:, :, 1].cuda()
        ctr_x   = (boxes[:, :, 0] + 0.5 * widths).cuda()
        ctr_y   = (boxes[:, :, 1] + 0.5 * heights).cuda()

        dx = (deltas[:, :, 0] * self.std[0] + self.mean[0]).cuda()
        dy = (deltas[:, :, 1] * self.std[1] + self.mean[1]).cuda()
        dw = (deltas[:, :, 2] * self.std[2] + self.mean[2]).cuda()
        dh = (deltas[:, :, 3] * self.std[3] + self.mean[3]).cuda()

        pred_ctr_x = (ctr_x + dx * widths).cuda()
        pred_ctr_y = (ctr_y + dy * heights).cuda()
        pred_w     = (torch.exp(dw) * widths).cuda()
        pred_h     = (torch.exp(dh) * heights).cuda()

        pred_boxes_x1 = (pred_ctr_x - 0.5 * pred_w).cuda()
        pred_boxes_y1 = (pred_ctr_y - 0.5 * pred_h).cuda()
        pred_boxes_x2 = (pred_ctr_x + 0.5 * pred_w).cuda()
        pred_boxes_y2 = (pred_ctr_y + 0.5 * pred_h).cuda()

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2).cuda()

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape
        # width = width.cuda()
        # height=height.cuda()
        # num_channels = num_channels.cuda()
        boxes[:, :, 0] = boxes[:, :, 0].cuda()
        boxes[:, :, 1] = boxes[:, :, 1].cuda()
        boxes[:, :, 2] = boxes[:, :, 2].cuda()
        boxes[:, :, 3] = boxes[:, :, 3].cuda()
        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0).cuda()
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0).cuda()

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width).cuda()
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height).cuda()
      
        return boxes
