"""C3D"""
import torch.nn as nn
import torch
import math
import torch.nn.functional as F


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()
    else:
        ft_module_names = []
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


class C3D(nn.Module):
    """C3D with BN and pool5 to be AdaptiveAvgPool3d(1)."""

    def __init__(self, proj_flag=False):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3a = nn.BatchNorm3d(256)
        self.relu3a = nn.ReLU()
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3b = nn.BatchNorm3d(256)
        self.relu3b = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4a = nn.BatchNorm3d(512)
        self.relu4a = nn.ReLU()
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4b = nn.BatchNorm3d(512)
        self.relu4b = nn.ReLU()
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5a = nn.BatchNorm3d(512)
        self.relu5a = nn.ReLU()
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5b = nn.BatchNorm3d(512)
        self.relu5b = nn.ReLU()

        self.pool5 = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.bn3a(x)
        x = self.relu3a(x)
        x = self.conv3b(x)
        x = self.bn3b(x)
        x = self.relu3b(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.bn4a(x)
        x = self.relu4a(x)
        x = self.conv4b(x)
        x = self.bn4b(x)
        x = self.relu4b(x)
        x = self.pool4(x)

        x = self.conv5a(x)
        x = self.bn5a(x)
        x = self.relu5a(x)
        x = self.conv5b(x)
        x = self.bn5b(x)
        x = self.relu5b(x)

        x = self.pool5(x)
        x = x.view(-1, 512)

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


class C3DBYOL(nn.Module):
    def __init__(self,
                 pretrain=True,
                 momentum=0.996,
                 **kwargs):
        super(C3DBYOL, self).__init__()
        if pretrain:
            self.momentum = momentum
            self.online_net = C3D(proj_flag=False)
            self.target_net = C3D(proj_flag=False)
            self.predictor = Predictor(dim=512, prediction_size=512, prediction_hidden_size=4096)
            self._set_grad(self.target_net, False)
            self.overlap_spa = nn.Linear(1024, 5)
            self.overlap_tem = nn.Linear(1024, 5)
            self.pb_cls = nn.Linear(512, 4)
            self.rotate_cls = nn.Linear(512, 4)
        else:
            self.online_net = C3D(proj_flag=False)
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
            online_feat_1 = self.online_net(x1)
            online_feat_2 = self.online_net(x2)
            online_feat_1_pred = self.predictor(online_feat_1)
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
    c3d = C3D()
