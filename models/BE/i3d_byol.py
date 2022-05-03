#!/usr/bin/env python
# -*- coding: utf-8 -*-
# For fair comparison, this code was modified from https://github.com/FingerRec/BE
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

import math
import os
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import copy


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    print("Layers to finetune : ", ft_module_names)

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            v.requires_grad = False
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class BNM(nn.Module):
    def __init__(self):
        super(BNM, self).__init__()

    def forward(self, x):
        out = -torch.norm(x, 'nuc')
        return out


def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init


class Unit3Dpy(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation='relu',
                 padding='SAME',
                 use_bias=False,
                 use_bn=True):
        super(Unit3Dpy, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == 'VALID':
            padding_shape = 0
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if padding == 'SAME':
            if not simplify_pad:
                self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    bias=use_bias)
            else:
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=pad_size,
                    bias=use_bias)
        elif padding == 'VALID':
            self.conv3d = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding_shape,
                stride=stride,
                bias=use_bias)
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if self.use_bn:
            self.batch3d = torch.nn.BatchNorm3d(out_channels)

        if activation == 'relu':
            self.activation = torch.nn.functional.relu

    def forward(self, inp):
        if self.padding == 'SAME' and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = torch.nn.functional.relu(out)
        return out


class MaxPool3dTFPadding(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)
        # self.pool = torch.nn.AvgPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class Mixed(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mixed, self).__init__()
        # Branch 0
        self.branch_0 = Unit3Dpy(
            in_channels, out_channels[0], kernel_size=(1, 1, 1))

        # Branch 1
        branch_1_conv1 = Unit3Dpy(
            in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Unit3Dpy(
            out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
        self.branch_1 = torch.nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Unit3Dpy(
            in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Unit3Dpy(
            out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
        self.branch_2 = torch.nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch3
        branch_3_pool = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
        branch_3_conv2 = Unit3Dpy(
            in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = torch.nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out


class I3D(torch.nn.Module):
    def __init__(self,
                 num_classes=0,
                 modality='rgb',
                 dropout_prob=0,
                 name='inception',
                 with_classifier=False,
                 projection=False
                 ):
        super(I3D, self).__init__()

        self.name = name
        self.num_classes = num_classes
        if modality == 'rgb':
            in_channels = 3
        elif modality == 'flow':
            in_channels = 2
        else:
            raise ValueError(
                '{} not among known modalities [rgb|flow]'.format(modality))
        self.modality = modality

        conv3d_1a_7x7 = Unit3Dpy(
            out_channels=64,
            in_channels=in_channels,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding='SAME')
        # 1st conv-pool
        self.conv3d_1a_7x7 = conv3d_1a_7x7
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        # conv conv
        conv3d_2b_1x1 = Unit3Dpy(
            out_channels=64,
            in_channels=64,
            kernel_size=(1, 1, 1),
            padding='SAME')
        self.conv3d_2b_1x1 = conv3d_2b_1x1
        conv3d_2c_3x3 = Unit3Dpy(
            out_channels=192,
            in_channels=64,
            kernel_size=(3, 3, 3),
            padding='SAME')
        self.conv3d_2c_3x3 = conv3d_2c_3x3 #here padding = 1 may influence the result
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')

        # Mixed_3b
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])

        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')

        # Mixed 4
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])

        self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')

        # Mixed 5
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

        # =========================end signal ======================================
        self.with_classifier = with_classifier
        if with_classifier:
            # if with_classifier:
            # self.avg_pool = torch.nn.AvgPool3d((2, 4, 4), (1, 1, 1))
            self.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))
            # self.avg_pool = torch.nn.AdaptiveAvgPool3d((1,1,1))
            self.dropout = torch.nn.Dropout(dropout_prob)
            self.conv3d_0c_1x1_custom = Unit3Dpy(
                in_channels=1024,
                out_channels=self.num_classes,
                kernel_size=(7, 1, 1),
                activation=None,
                use_bias=False,
                use_bn=False)
        else:
            # MLP improve performance
            self.projection = projection
            if projection:
                print("no classifier, projection")
                self.id_head = nn.Sequential(
                                            # torch.nn.Dropout3d(0.5),
                                            # torch.nn.AdaptiveAvgPool3d((2,1,1)),
                                            #  Unit3Dpy(in_channels=1024, out_channels=1024,
                                            #      kernel_size=(2, 1, 1), activation=None, use_bias=False, use_bn=False),
                                            torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
                                            Flatten(),
                                            # torch.nn.Linear(1024, 256),
                                            # nn.ReLU(),
                                            torch.nn.Linear(1024, 128),
                                            #Sharpen(),
                                            #BNM(),
                                            Normalize(2)
                                            )
            else:
                print("No classifier, No projection")
                self.id_head = nn.Sequential(
                                            # torch.nn.Dropout3d(0.5),
                                            # torch.nn.AdaptiveAvgPool3d((2,1,1)),
                                            #  Unit3Dpy(in_channels=1024, out_channels=1024,
                                            #      kernel_size=(2, 1, 1), activation=None, use_bias=False, use_bn=False),
                                            torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
                                            Flatten(),
                                            # torch.nn.Linear(1024, 256),
                                            # nn.ReLU(),
                                            # torch.nn.Linear(1024, 128),
                                            # Sharpen(),
                                            # BNM(),
                                            Normalize(2)
                                            )
        # # ===========================Motion Enhance Module =============================
        # self.motion_enhance_module = MotionEnhance()
        # # ===========================Binary classification layers ======================
        # self.binary_classifier = binary_classifier
        # if binary_classifier:
        #     # self.projection = nn.Sequential(torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1)),
        #     #                                 Unit3Dpy(
        #     #                                     in_channels=1024,
        #     #                                     out_channels=256,
        #     #                                     kernel_size=(7, 1, 1),
        #     #                                     activation=None,
        #     #                                     use_bias=False,
        #     #                                     use_bn=False),
        #     #                                 nn.ReLU(),
        #     #                                 nn.BatchNorm3d(256),
        #     #                                 nn.Conv3d(256, 6, 1, 1, bias=False),
        #     #                                 Flatten()
        #     #                                 # torch.nn.Softmax(1)
        #     #                                 )
        #     self.num_pool_features = 2 * 7 * 7 * 256
        #     self.cls_num_classes = 5
        #     self.projection = nn.Sequential(
        #                                     Unit3Dpy(
        #                                         in_channels=1024,
        #                                         out_channels=256,
        #                                         kernel_size=(1, 1, 1),
        #                                         activation=None,
        #                                         use_bias=False,
        #                                         use_bn=False),
        #                                     Flatten(),
        #                                     nn.Linear(self.num_pool_features, 512, bias=False),
        #                                     nn.BatchNorm1d(512),
        #                                     nn.ReLU(inplace=True),
        #                                     nn.Linear(512, 512, bias=False),
        #                                     nn.BatchNorm1d(512),
        #                                     nn.ReLU(inplace=True),
        #                                     nn.Linear(512, self.cls_num_classes)
        #                                   )

    def forward(self, inp):
        # print(inp.size())
        # features = []
        out = self.conv3d_1a_7x7(inp)
        out = self.maxPool3d_2a_3x3(out)
        # if motion_enhance:
        #     out = self.motion_enhance_module(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        # features.append(out)
        out = self.maxPool3d_3a_3x3(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        # features.append(out)
        out = self.maxPool3d_4a_3x3(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        # features.append(out)
        feature_map = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(feature_map)
        out = self.mixed_5c(out)
        if self.with_classifier:
            out = self.avg_pool(out)
            out = self.dropout(out)
            out = self.conv3d_0c_1x1_custom(out)
            out = out.squeeze(3)
            out = out.squeeze(3)
            out = out.mean(2)
            return out
        else:
            out = self.id_head(out)
            return out
        # features.append(out)
        # if self.binary_classifier:
        #     return self.projection(out)
        # if return_conv:
        #     # return features
        #     return out
        # if not self.with_classifier:
        #     id_out = self.id_head(out)
        #     return id_out, out

        # return F.log_softmax(out, dim=1)

    def load_tf_weights(self, sess):
        state_dict = {}
        if self.modality == 'rgb':
            prefix = 'RGB/inception_i3d'
        elif self.modality == 'flow':
            prefix = 'Flow/inception_i3d'
        load_conv3d(state_dict, 'conv3d_1a_7x7', sess,
                    os.path.join(prefix, 'Conv3d_1a_7x7'))
        load_conv3d(state_dict, 'conv3d_2b_1x1', sess,
                    os.path.join(prefix, 'Conv3d_2b_1x1'))
        load_conv3d(state_dict, 'conv3d_2c_3x3', sess,
                    os.path.join(prefix, 'Conv3d_2c_3x3'))

        load_mixed(state_dict, 'mixed_3b', sess,
                   os.path.join(prefix, 'Mixed_3b'))
        load_mixed(state_dict, 'mixed_3c', sess,
                   os.path.join(prefix, 'Mixed_3c'))
        load_mixed(state_dict, 'mixed_4b', sess,
                   os.path.join(prefix, 'Mixed_4b'))
        load_mixed(state_dict, 'mixed_4c', sess,
                   os.path.join(prefix, 'Mixed_4c'))
        load_mixed(state_dict, 'mixed_4d', sess,
                   os.path.join(prefix, 'Mixed_4d'))
        load_mixed(state_dict, 'mixed_4e', sess,
                   os.path.join(prefix, 'Mixed_4e'))
        # Here goest to 0.1 max error with tf
        load_mixed(state_dict, 'mixed_4f', sess,
                   os.path.join(prefix, 'Mixed_4f'))

        load_mixed(
            state_dict,
            'mixed_5b',
            sess,
            os.path.join(prefix, 'Mixed_5b'),
            fix_typo=True)
        load_mixed(state_dict, 'mixed_5c', sess,
                   os.path.join(prefix, 'Mixed_5c'))
        load_conv3d(
            state_dict,
            'conv3d_0c_1x1',
            sess,
            os.path.join(prefix, 'Logits', 'Conv3d_0c_1x1'),
            bias=True,
            bn=False)
        self.load_state_dict(state_dict)


def get_conv_params(sess, name, bias=False):
    # Get conv weights
    conv_weights_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'w:0'))
    if bias:
        conv_bias_tensor = sess.graph.get_tensor_by_name(
            os.path.join(name, 'b:0'))
        conv_bias = sess.run(conv_bias_tensor)
    conv_weights = sess.run(conv_weights_tensor)
    conv_shape = conv_weights.shape

    kernel_shape = conv_shape[0:3]
    in_channels = conv_shape[3]
    out_channels = conv_shape[4]

    conv_op = sess.graph.get_operation_by_name(
        os.path.join(name, 'convolution'))
    padding_name = conv_op.get_attr('padding')
    padding = _get_padding(padding_name, kernel_shape)
    all_strides = conv_op.get_attr('strides')
    strides = all_strides[1:4]
    conv_params = [
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding
    ]
    if bias:
        conv_params.append(conv_bias)
    return conv_params


def get_bn_params(sess, name):
    moving_mean_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'moving_mean:0'))
    moving_var_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'moving_variance:0'))
    beta_tensor = sess.graph.get_tensor_by_name(os.path.join(name, 'beta:0'))
    moving_mean = sess.run(moving_mean_tensor)
    moving_var = sess.run(moving_var_tensor)
    beta = sess.run(beta_tensor)
    return moving_mean, moving_var, beta


def _get_padding(padding_name, conv_shape):
    padding_name = padding_name.decode("utf-8")
    if padding_name == "VALID":
        return [0, 0]
    elif padding_name == "SAME":
        # return [math.ceil(int(conv_shape[0])/2), math.ceil(int(conv_shape[1])/2)]
        return [
            math.floor(int(conv_shape[0]) / 2),
            math.floor(int(conv_shape[1]) / 2),
            math.floor(int(conv_shape[2]) / 2)
        ]
    else:
        raise ValueError('Invalid padding name ' + padding_name)


def load_conv3d(state_dict, name_pt, sess, name_tf, bias=False, bn=True):
    # Transfer convolution params
    conv_name_tf = os.path.join(name_tf, 'conv_3d')
    conv_params = get_conv_params(sess, conv_name_tf, bias=bias)
    if bias:
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding, conv_bias = conv_params
    else:
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding = conv_params

    conv_weights_rs = np.transpose(
        conv_weights, (4, 3, 0, 1,
                       2))  # to pt format (out_c, in_c, depth, height, width)
    state_dict[name_pt + '.conv3d.weight'] = torch.from_numpy(conv_weights_rs)
    if bias:
        state_dict[name_pt + '.conv3d.bias'] = torch.from_numpy(conv_bias)

    # Transfer batch norm params
    if bn:
        conv_tf_name = os.path.join(name_tf, 'batch_norm')
        moving_mean, moving_var, beta = get_bn_params(sess, conv_tf_name)

        out_planes = conv_weights_rs.shape[0]
        state_dict[name_pt + '.batch3d.weight'] = torch.ones(out_planes)
        state_dict[name_pt +
                   '.batch3d.bias'] = torch.from_numpy(beta.squeeze())
        state_dict[name_pt
                   + '.batch3d.running_mean'] = torch.from_numpy(moving_mean.squeeze())
        state_dict[name_pt
                   + '.batch3d.running_var'] = torch.from_numpy(moving_var.squeeze())


def load_mixed(state_dict, name_pt, sess, name_tf, fix_typo=False):
    # Branch 0
    load_conv3d(state_dict, name_pt + '.branch_0', sess,
                os.path.join(name_tf, 'Branch_0/Conv3d_0a_1x1'))

    # Branch .1
    load_conv3d(state_dict, name_pt + '.branch_1.0', sess,
                os.path.join(name_tf, 'Branch_1/Conv3d_0a_1x1'))
    load_conv3d(state_dict, name_pt + '.branch_1.1', sess,
                os.path.join(name_tf, 'Branch_1/Conv3d_0b_3x3'))

    # Branch 2
    load_conv3d(state_dict, name_pt + '.branch_2.0', sess,
                os.path.join(name_tf, 'Branch_2/Conv3d_0a_1x1'))
    if fix_typo:
        load_conv3d(state_dict, name_pt + '.branch_2.1', sess,
                    os.path.join(name_tf, 'Branch_2/Conv3d_0a_3x3'))
    else:
        load_conv3d(state_dict, name_pt + '.branch_2.1', sess,
                    os.path.join(name_tf, 'Branch_2/Conv3d_0b_3x3'))

    # Branch 3
    load_conv3d(state_dict, name_pt + '.branch_3.1', sess,
                os.path.join(name_tf, 'Branch_3/Conv3d_0b_1x1'))


class Projector(nn.Module):
    def __init__(self, dim, projection_size, projection_hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, projection_hidden_size),
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
            nn.Linear(dim, prediction_hidden_size),
            nn.BatchNorm1d(prediction_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(prediction_hidden_size, prediction_size)
        )

    def forward(self, x):
        return self.net(x)


class I3DBYOL(nn.Module):
    def __init__(self, momentum=0.996, pretrain=True, opts=None):
        super(I3DBYOL, self).__init__()
        if pretrain:
            self.momentum = momentum
            self.online_net = I3D(with_classifier=False, projection=False)
            self.target_net = copy.deepcopy(self.online_net)
            self.predictor = Predictor(dim=1024, prediction_size=1024, prediction_hidden_size=4096)
            self._set_grad(self.target_net, False)
            self.overlap_spa = nn.Linear(2048, 5)
            self.overlap_tem = nn.Linear(2048, 5)
            self.pb_cls = nn.Linear(1024, 4)
            self.rot_cls = nn.Linear(1024, 4)
        else:
            self.online_net = I3D(num_classes=opts.n_classes, with_classifier=True, projection=False)

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
            out = self.online_net(x1)
            # if self.cls_bn:
            #     online_feat = F.normalize(online_feat, p=2, dim=1)
            #     online_feat = self.classify_bn(online_feat)
            #     out = self.classify(online_feat)
            # else:
            #     out = self.classify(online_feat)
            return out
        elif o_type == 'scratch':
            online_feat = self.online_net(x1)
            out = self.classify(online_feat)
            return out
