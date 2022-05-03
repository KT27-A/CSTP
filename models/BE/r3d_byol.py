# For fair comparison, this code was modified from https://github.com/FingerRec/BE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import copy

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    if ft_begin_index <= 4:
        for i in range(ft_begin_index, 5):
            ft_module_names.append('layer{}'.format(i))
        ft_module_names.append('classify')
    else:
        ft_module_names.append('classify')

    print("Modules to finetune : ", ft_module_names)

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                print("Layers to finetune : ", k)
                parameters.append({'params': v})
                break
        else:
            v.requires_grad = False
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
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
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
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


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 sample_size=224,
                 sample_duration=16,
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
        x = self.avgpool(x)
        x = x.view(-1, 512)

        return x


class Projector(nn.Module):
    def __init__(self, dim, projection_size, projection_hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, projection_hidden_size),
            nn.BatchNorm1d(projection_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, dim, prediction_size, prediction_hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, prediction_hidden_size),
            nn.BatchNorm1d(prediction_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(prediction_hidden_size, prediction_size)
        )

    def forward(self, x):
        return self.net(x)


class R3DBYOL(nn.Module):
    def __init__(self, momentum=0.996, pretrain=True, cls_bn=False, opts=None):
        super(R3DBYOL, self).__init__()
        if pretrain:
            self.momentum = momentum
            self.online_net = globals()['resnet{}'.format(opts.model_depth)](
                                        sample_size=opts.sample_size,
                                        sample_duration=opts.sample_duration,
                                        shortcut_type=opts.sc_type,
                                        num_classes=opts.n_classes)
            self.target_net = copy.deepcopy(self.online_net)
            self.predictor = Predictor(dim=512, prediction_size=512, prediction_hidden_size=4096)
            self._set_grad(self.target_net, False)
            self.overlap_spa = nn.Linear(1024, 5)
            self.overlap_tem = nn.Linear(1024, 5)
            self.pb_cls = nn.Linear(512, 4)
            self.rot_cls = nn.Linear(512, 4)
        else:
            self.online_net = globals()['resnet{}'.format(opts.model_depth)](
                                        sample_size=opts.sample_size,
                                        sample_duration=opts.sample_duration,
                                        shortcut_type=opts.sc_type,
                                        num_classes=opts.n_classes)
            self.cls_bn = cls_bn
            if self.cls_bn:
                self.classify_bn = nn.BatchNorm1d(512)
            self.classify = nn.Linear(512, opts.n_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = self._glorot_uniform(m.weight)
            elif isinstance(m, nn.Conv3d):
                m.weight = self._glorot_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight = self._glorot_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight = self._glorot_uniform(m.weight)

    def _calculate_fan_in_and_fan_out(self, tensor):
        dimensions = tensor.dim()
        if dimensions < 2:
            return int(tensor.size(0)/2), int(tensor.size(0)/2)
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        else:
            receptive_field_size = 1
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
        return fan_in, fan_out

    @torch.no_grad()
    def _glorot_uniform(self, tensor):
        fan_in, fan_out = self._calculate_fan_in_and_fan_out(tensor)
        std = math.sqrt(6.0 / float(fan_in + fan_out))
        return tensor.uniform_(-std, std)

    def _update_target_net(self):
        """
        Momentum update of the key net
        """
        for param_q, param_k in zip(self.online_net.parameters(), self.target_net.parameters()):
            # what in the paper
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

            # what adjust from the github code
            # param_k.data = param_q

    def _set_grad(self, model, val):
        for p in model.parameters():
            p.requires_grad = val

    def _loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def _cal_loss(self, online_feat_1, online_feat_2, target_feat_1, target_feat_2):
        loss_one = self._loss_fn(online_feat_1, target_feat_2)
        loss_two = self._loss_fn(target_feat_1, online_feat_2)
        loss = loss_one + loss_two
        return loss

    @torch.no_grad()
    def _concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        '''
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        '''
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self._concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        '''
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        '''
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self._concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, x1, x2=None, o_type='r_byol'):
        if o_type == "loss_com":
            online_feat_1 = self.online_net(x1)
            online_feat_1_pred = self.predictor(online_feat_1)
            online_feat_2 = self.online_net(x2)
            online_feat_2_pred = self.predictor(online_feat_2)
            with torch.no_grad():
                self._update_target_net()
                target_feat_1 = self.target_net(x1)
                target_feat_2 = self.target_net(x2)
                # target_feat_1.detach_()
                # target_feat_2.detach_()
                target_feat_1 = target_feat_1.detach()
                target_feat_2 = target_feat_2.detach()
            loss = self._cal_loss(online_feat_1_pred, online_feat_2_pred,
                                  target_feat_1, target_feat_2)

            feat_cat = torch.cat((online_feat_1, online_feat_2), dim=1)
            pred_spa = self.overlap_spa(feat_cat)
            pred_tem = self.overlap_tem(feat_cat)
            pred_pb_1 = self.pb_cls(online_feat_1)
            pred_pb_2 = self.pb_cls(online_feat_2)
            pred_rot_1 = self.rot_cls(online_feat_1)
            pred_rot_2 = self.rot_cls(online_feat_2)

            return loss.mean(), (pred_spa, pred_tem, pred_pb_1, pred_pb_2, pred_rot_1, pred_rot_2)

        if o_type == 'r_byol':
            online_feat_1 = self.online_net(x1)
            online_feat_1 = self.predictor(online_feat_1)
            if self.shuffle_bn:
                x2, self.idx_unshuffle = self._batch_shuffle_ddp(x2)
            online_feat_2 = self.online_net(x2)
            online_feat_2 = self.predictor(online_feat_2)
            with torch.no_grad():
                self._update_target_net()
                target_feat_1 = self.target_net(x1)
                target_feat_2 = self.target_net(x2)
            loss = self._cal_loss(online_feat_1, online_feat_2, target_feat_1.detach(), target_feat_2.detach())
            return loss.mean()
        elif o_type in ['ft_fc', 'ft_all', 'test']:
            online_feat = self.online_net(x1)
            if self.cls_bn:
                online_feat = F.normalize(online_feat, p=2, dim=1)
                online_feat = self.classify_bn(online_feat)
                out = self.classify(online_feat)
            else:
                out = self.classify(online_feat)
            return out
        elif o_type == 'scratch':
            online_feat = self.online_net(x1)
            out = self.classify(online_feat)
            return out


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
