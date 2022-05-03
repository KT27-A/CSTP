# modified from https://raw.githubusercontent.com/qijiezhao/s3d.pytorch/master/S3DG_Pytorch.py 
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

# pytorch default: torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# tensorflow s3d code: torch.nn.BatchNorm3d(num_features, eps=1e-3, momentum=0.001, affine=True, track_running_stats=True)


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


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, 
                              kernel_size=kernel_size, stride=stride, 
                              padding=padding, bias=False)

        # self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        # init
        self.conv.weight.data.normal_(mean=0, std=0.01)  # original s3d is truncated normal within 2 std
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class STConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(STConv3d, self).__init__()
        if isinstance(stride, tuple):
            t_stride = stride[0]
            stride = stride[-1]
        else:  # int
            t_stride = stride

        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(1, kernel_size, kernel_size),
                               stride=(1, stride, stride), padding=(0, padding, padding), bias=False)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=(kernel_size, 1, 1),
                               stride=(t_stride, 1, 1), padding=(padding, 0, 0), bias=False)

        # self.bn1=nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        # self.bn2=nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        # init
        self.conv1.weight.data.normal_(mean=0, std=0.01)  # original s3d is truncated normal within 2 std
        self.conv2.weight.data.normal_(mean=0, std=0.01)  # original s3d is truncated normal within 2 std
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class SelfGating(nn.Module):
    def __init__(self, input_dim):
        super(SelfGating, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, input_tensor):
        """Feature gating as used in S3D-G"""
        spatiotemporal_average = torch.mean(input_tensor, dim=[2, 3, 4])
        weights = self.fc(spatiotemporal_average)
        weights = torch.sigmoid(weights)
        return weights[:, :, None, None, None] * input_tensor


class SepInception(nn.Module):
    def __init__(self, in_planes, out_planes, gating=False):
        super(SepInception, self).__init__()

        assert len(out_planes) == 6
        assert isinstance(out_planes, list)

        [num_out_0_0a,
         num_out_1_0a, num_out_1_0b,
         num_out_2_0a, num_out_2_0b,
         num_out_3_0b] = out_planes

        self.branch0 = nn.Sequential(
            BasicConv3d(in_planes, num_out_0_0a, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(in_planes, num_out_1_0a, kernel_size=1, stride=1),
            STConv3d(num_out_1_0a, num_out_1_0b, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(in_planes, num_out_2_0a, kernel_size=1, stride=1),
            STConv3d(num_out_2_0a, num_out_2_0b, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(in_planes, num_out_3_0b, kernel_size=1, stride=1),
        )

        self.out_channels = sum([num_out_0_0a, num_out_1_0b, num_out_2_0b, num_out_3_0b])

        self.gating = gating
        if gating:
            self.gating_b0 = SelfGating(num_out_0_0a)
            self.gating_b1 = SelfGating(num_out_1_0b)
            self.gating_b2 = SelfGating(num_out_2_0b)
            self.gating_b3 = SelfGating(num_out_3_0b)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        if self.gating:
            x0 = self.gating_b0(x0)
            x1 = self.gating_b1(x1)
            x2 = self.gating_b2(x2)
            x3 = self.gating_b3(x3)

        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class S3D(nn.Module):
    def __init__(self, input_channel=3, gating=False, slow=False, proj_flag=False):
        super(S3D, self).__init__()
        self.gating = gating
        self.slow = slow

        if slow:
            self.Conv_1a = STConv3d(input_channel, 64, kernel_size=7, stride=(1, 2, 2), padding=3)
        else:  # normal
            self.Conv_1a = STConv3d(input_channel, 64, kernel_size=7, stride=2, padding=3)

        self.block1 = nn.Sequential(self.Conv_1a)  # (64, 32, 112, 112)

        ###################################

        self.MaxPool_2a = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.Conv_2b = BasicConv3d(64, 64, kernel_size=1, stride=1)
        self.Conv_2c = STConv3d(64, 192, kernel_size=3, stride=1, padding=1)

        self.block2 = nn.Sequential(
            self.MaxPool_2a,  # (64, 32, 56, 56)
            self.Conv_2b,  # (64, 32, 56, 56)
            self.Conv_2c)  # (192, 32, 56, 56)

        ###################################

        self.MaxPool_3a = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.Mixed_3b = SepInception(in_planes=192, out_planes=[64, 96, 128, 16, 32, 32], gating=gating)
        self.Mixed_3c = SepInception(in_planes=256, out_planes=[128, 128, 192, 32, 96, 64], gating=gating)

        self.block3 = nn.Sequential(
            self.MaxPool_3a,    # (192, 32, 28, 28)
            self.Mixed_3b,      # (256, 32, 28, 28)
            self.Mixed_3c)      # (480, 32, 28, 28)

        ###################################

        self.MaxPool_4a = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.Mixed_4b = SepInception(in_planes=480, out_planes=[192, 96, 208, 16, 48, 64], gating=gating)
        self.Mixed_4c = SepInception(in_planes=512, out_planes=[160, 112, 224, 24, 64, 64], gating=gating)
        self.Mixed_4d = SepInception(in_planes=512, out_planes=[128, 128, 256, 24, 64, 64], gating=gating)
        self.Mixed_4e = SepInception(in_planes=512, out_planes=[112, 144, 288, 32, 64, 64], gating=gating)
        self.Mixed_4f = SepInception(in_planes=528, out_planes=[256, 160, 320, 32, 128, 128], gating=gating)

        self.block4 = nn.Sequential(
            self.MaxPool_4a,  # (480, 16, 14, 14)
            self.Mixed_4b,    # (512, 16, 14, 14)
            self.Mixed_4c,    # (512, 16, 14, 14)
            self.Mixed_4d,    # (512, 16, 14, 14)
            self.Mixed_4e,    # (528, 16, 14, 14)
            self.Mixed_4f)    # (832, 16, 14, 14)

        ###################################

        self.MaxPool_5a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
        self.Mixed_5b = SepInception(in_planes=832, out_planes=[256, 160, 320, 32, 128, 128], gating=gating)
        self.Mixed_5c = SepInception(in_planes=832, out_planes=[384, 192, 384, 48, 128, 128], gating=gating)

        self.block5 = nn.Sequential(
            self.MaxPool_5a,  # (832, 8, 7, 7)
            self.Mixed_5b,    # (832, 8, 7, 7)
            self.Mixed_5c)    # (1024, 8, 7, 7)

        self.avgpooling = nn.AdaptiveAvgPool3d((1, 1, 1))

        # projector
        self.proj_flag = proj_flag
        if self.proj_flag:
            self.project = Projector(dim=1024, projection_size=1024, projection_hidden_size=1024)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpooling(x)
        x = x.view(-1, 1024)
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


class S3DClassify(nn.Module):
    def __init__(self, momentum=0.996, pretrain=True, classify_bn=True, shuffle_bn=False, **kwargs):
        super(S3DClassify, self).__init__()
        self.online_net = S3D(gating=kwargs['gating'], slow=kwargs['slow'], linear_flag='no_proj')
        self.classify = nn.Linear(1024, kwargs['num_classes'])
        #if classify_bn:
        #    print('classify_bn is true, Feature norm and Batch norm on final features')
        #    self.classify_bn = nn.BatchNorm1d(1024)
        #    self.l2_norm = True
        #else:
        #    self.l2_norm = False
        self.l2_norm = False
        self.classify = nn.Sequential(nn.Linear(1024, 1024),
                                      nn.BatchNorm1d(1024),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(1024, kwargs['num_classes'])) 
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

    def forward(self, x1, x2=None, o_type='r_byol'):
        online_feat = self.online_net(x1)
        if self.l2_norm:
            online_feat = F.normalize(online_feat, p=2, dim=1)
            online_feat = self.classify_bn(online_feat)
            out = self.classify(online_feat)
        else:
            out = self.classify(online_feat)
        return out


class S3DGBYOL(nn.Module):
    def __init__(self, momentum=0.996, pretrain=True, classify_bn=True, shuffle_bn=False, **kwargs):
        super(S3DGBYOL, self).__init__()
        if pretrain:
            self.momentum = momentum
            self.online_net = S3D(gating=kwargs['gating'], slow=kwargs['slow'], proj_flag=True)
            self.target_net = S3D(gating=kwargs['gating'], slow=kwargs['slow'], proj_flag=True)
            self.predictor = Predictor(dim=1024, prediction_size=1024, prediction_hidden_size=4096)
            self._set_grad(self.target_net, False)
            self.overlap_spa = nn.Sequential(nn.Linear(2048, 2048),
                                             nn.BatchNorm1d(2048),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(2048, 5))
            self.overlap_tem = nn.Sequential(nn.Linear(2048, 2048),
                                             nn.BatchNorm1d(2048),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(2048, 5))
            self.pb_cls = nn.Sequential(nn.Linear(1024, 1024),
                                        nn.BatchNorm1d(1024),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(1024, 5))
            self.rotate_cls = nn.Sequential(nn.Linear(1024, 1024),
                                        nn.BatchNorm1d(1024),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(1024, 5))
        else:
            self.online_net = S3D(gating=kwargs['gating'], slow=kwargs['slow'], proj_flag=False)
            self.classify = nn.Linear(1024, kwargs['num_classes'])
            if classify_bn:
                print('classify_bn is true, Feature norm and Batch norm on final features')
                self.classify_bn = nn.BatchNorm1d(1024)
                self.l2_norm = True
            else:
                self.l2_norm = False

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
            if self.l2_norm:
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


if __name__ == '__main__':
    model = S3DGBYOL(gating=False, slow=False, num_classes=400)
    data = torch.FloatTensor(4, 3, 16, 224, 224)
    out = model(data, data, o_type='r_byol')
