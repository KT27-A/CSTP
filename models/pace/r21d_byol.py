"""R2plus1D"""
import math
import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple
import numpy as np
import torch.nn.functional as F


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


class SpatioTemporalConv(nn.Module):
    """Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, first_conv=False):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
        spatial_stride = (1, stride[1], stride[2])
        spatial_padding = (0, padding[1], padding[2])

        temporal_kernel_size = (kernel_size[0], 1, 1)
        temporal_stride = (stride[0], 1, 1)
        temporal_padding = (padding[0], 0, 0)

        # compute the number of intermediary channels (M) using formula
        # from the paper section 3.5
        intermed_channels = int(
            math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) / \
                       (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))
        # print(intermed_channels)

        # the spatial conv is effectively a 2D conv due to the
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm
        # and ReLU added inside the model constructor, not here. This is an
        # intentional design choice, to allow this module to externally act
        # identical to a standard Conv3D, so it can be reused easily in any
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                       stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x


class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2 when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
        return x


class R2Plus1DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.
        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers.
    """

    def __init__(self, layer_sizes=(1, 1, 1, 1),
                 block_type=SpatioTemporalResBlock, proj_flag=False):
        super(R2Plus1DNet, self).__init__()

        # first conv, with stride 1x2x2 and kernel size 1x7x7
        self.conv1 = SpatioTemporalConv(3, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU(inplace=True)
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj_flag = proj_flag
        if self.proj_flag:
            self.project = Projector(dim=512, projection_size=512, projection_hidden_size=4096)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool(x)
        x = x.view(-1, 512)

        if self.proj_flag:
            x_proj = self.project(x)
            return x, x_proj
        else:
            return x


class Projector(nn.Module):
    def __init__(self, dim, projection_size, projection_hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, projection_hidden_size),
            nn.BatchNorm1d(projection_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, dim, prediction_size, prediction_hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, prediction_hidden_size),
            nn.BatchNorm1d(prediction_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(prediction_hidden_size, prediction_size)
        )

    def forward(self, x):
        return self.net(x)


class R21DBYOL(nn.Module):
    def __init__(self,
                 pretrain=True,
                 momentum=0.996,
                 **kwargs):
        super(R21DBYOL, self).__init__()
        if pretrain:
            self.momentum = momentum
            self.online_net = R2Plus1DNet(layer_sizes=(1, 1, 1, 1), proj_flag=True)
            self.target_net = R2Plus1DNet(layer_sizes=(1, 1, 1, 1), proj_flag=True)
            self.predictor = Predictor(dim=512, prediction_size=512, prediction_hidden_size=4096)
            self._set_grad(self.target_net, False)
            # self.overlap_spa = nn.Linear(1024, 5)
            # self.overlap_tem = nn.Linear(1024, 5)
            # self.pb_cls = nn.Linear(512, 4)
            # self.rotate_cls = nn.Linear(512, 4)
            self.overlap_spa = nn.Sequential(nn.Linear(1024, 1024),
                                             nn.BatchNorm1d(1024),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(1024, 5))
            self.overlap_tem = nn.Sequential(nn.Linear(1024, 1024),
                                             nn.BatchNorm1d(1024),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(1024, 5))
            self.pb_cls = nn.Sequential(nn.Linear(512, 512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, 5))
            self.rotate_cls = nn.Sequential(nn.Linear(512, 512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, 5))

        else:
            self.online_net = R2Plus1DNet(layer_sizes=(1, 1, 1, 1), proj_flag=False)
            self.classify = nn.Linear(512, kwargs["num_classes"])
            self.cls_bn = kwargs["cls_bn"]
            if self.cls_bn:
                print('classify_bn is true, Feature norm and Batch norm on final features')
                self.cls_bn = nn.BatchNorm1d(512)

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
        loss_two = self._loss_fn(online_feat_2, target_feat_1)
        loss = loss_one + loss_two
        return loss

    def forward(self, x1, x2=None, o_type=None):
        if o_type == "loss_com":
            online_feat_1, online_feat_1_proj = self.online_net(x1)
            online_feat_2, online_feat_2_proj = self.online_net(x2)
            online_feat_1_pred = self.predictor(online_feat_1_proj)
            online_feat_2_pred = self.predictor(online_feat_2_proj)
            with torch.no_grad():
                self._update_target_net()
                _, target_feat_1_proj = self.target_net(x1)
                _, target_feat_2_proj = self.target_net(x2)
                # target_feat_1.detach_()
                # target_feat_2.detach_()
                target_feat_1_proj = target_feat_1_proj.detach()
                target_feat_2_proj = target_feat_2_proj.detach()
            loss = self._cal_loss(online_feat_1_pred, online_feat_2_pred,
                                  target_feat_1_proj, target_feat_2_proj)
            # online_feat_1 = F.normalize(online_feat_1, p=2, dim=1)
            feat_cat = torch.cat((online_feat_1, online_feat_2), dim=1)
            pred_spa = self.overlap_spa(feat_cat)
            pred_tem = self.overlap_tem(feat_cat)
            pred_pb_1 = self.pb_cls(online_feat_1)
            pred_pb_2 = self.pb_cls(online_feat_2)
            pred_rot_1 = self.rotate_cls(online_feat_1)
            pred_rot_2 = self.rotate_cls(online_feat_2)

            return loss.mean(), (pred_spa, pred_tem, pred_pb_1, pred_pb_2, pred_rot_1, pred_rot_2)
        elif o_type == 'r_byol':
            online_feat_1 = self.online_net(x1)
            online_feat_1 = self.predictor(online_feat_1)
            online_feat_2 = self.online_net(x2)
            online_feat_2 = self.predictor(online_feat_2)
            with torch.no_grad():
                self._update_target_net()
                target_feat_1 = self.target_net(x1)
                target_feat_2 = self.target_net(x2)
            loss = self._cal_loss(online_feat_1, online_feat_2, target_feat_1, target_feat_2)
            return loss
        elif o_type in ['ft_fc', "ft_all", "test"]:
            online_feat = self.online_net(x1)
            online_feat = F.normalize(online_feat, p=2, dim=1)
            online_feat = self.cls_bn(online_feat)
            out = self.classify(online_feat)
            return out
        else:
            raise ValueError("Output cls is not exist!")


if __name__ == '__main__':
    r21d = R2Plus1DNet((1, 1, 1, 1))
