"""C3D"""
import torch.nn as nn


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

    def __init__(self, num_classes=4):
        super(C3D, self).__init__()
        self.num_classes = num_classes

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
        self.classify = nn.Linear(512, self.num_classes)

    def _extract(self, x):
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

    def forward(self, x1, x2, o_type='ctr'):
        if o_type == 'ctr':
            x1 = self._extract(x1)
            x2 = self._extract(x2)
            return x1, x2
        elif o_type == 'cls':
            x1 = self._extract(x1)
            x1 = self.classify(x1)
            return x1


if __name__ == '__main__':
    c3d = C3D()
