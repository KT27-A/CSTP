from data_process.datasets import UcfFineTune, UcfFineTuneLMDB, Kin400FTOfflineLMDB
from torch.utils.data import Dataset, DataLoader
import getpass
import os
import socket
import numpy as np
from PIL import Image, ImageFilter
import argparse
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from models.model import generate_model
from opts import parse_opts
from torch.autograd import Variable
import time
import sys
from utils import AverageMeter
from data_process.preprocess_data import get_transforms
import glob


if __name__ == "__main__":
    # print configuration options
    opts = parse_opts()
    if torch.cuda.is_available():
        opts.cuda = True
    opts.distributed = False
    opts.device = torch.device("cuda:0" if opts.cuda else 'cpu')
    print(opts)
    opts.arch = '{}-{}'.format(opts.model_name, opts.model_depth)

    test_transform = get_transforms(mode=opts.transform_mode, opts=opts)
    print("Preprocessing testing data ...")
    test_data = globals()['{}'.format(opts.dataset)](data_type=opts.task,
                                                     opts=opts,
                                                     split=opts.split,
                                                     sp_transform=test_transform)
    print("Length of testing data = ", len(test_data))

    print("Preparing datatloaders ...")
    test_dataloader = DataLoader(test_data,
                                 batch_size=opts.batch_size,
                                 shuffle=False,
                                 num_workers=opts.n_workers,
                                 pin_memory=True,
                                 drop_last=False)
    print("Length of test datatloader = ", len(test_dataloader))

    if not opts.test_md_path:
        opts.test_md_path = glob.glob(os.path.join(opts.result_path, opts.dataset, opts.t_ft_task, "*_max.pth"))
        if len(opts.test_md_path) > 1:
            raise ValueError("Too many models in result path")
        else:
            opts.test_md_path = opts.test_md_path[0]

    # Loading model and checkpoint
    model = generate_model(opts)
    accuracies = AverageMeter()
    clip_accuracies = AverageMeter()

    # Path to store results
    result_path = "{}/{}/".format(opts.result_path, opts.dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    f = open(os.path.join(result_path, "test_{}{}_{}_{}_{}_{}_plusone.txt".format(opts.model_name,
                                                                                  opts.model_depth,
                                                                                  opts.dataset,
                                                                                  opts.split,
                                                                                  opts.modality,
                                                                                  opts.sample_duration)), 'w+')
    f.write(str(opts) + "\n")
    f.flush()

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_dataloader):
            inputs = torch.squeeze(inputs, 0)
            inputs = inputs.to(opts.device, non_blocking=True)
            labels = labels.to(opts.device, non_blocking=True)
            outputs = model(inputs, None, o_type=opts.task)
            pred5 = np.array(torch.mean(outputs, dim=0, keepdim=True).topk(5, 1, True)[1].cpu().data[0])
            acc = float(pred5[0] == labels[0])
            accuracies.update(acc, 1)
            line = 'Video[{}]:\ttop5 = {}\ttop1 = {}\tgt = {}\tacc = {}'.format(i,
                                                                                pred5,
                                                                                pred5[0],
                                                                                labels[0],
                                                                                accuracies.avg)
            print(line)
            f.write(line + '\n')
            f.flush()

    print("Video accuracy = ", accuracies.avg)
    line = "Video accuracy = " + str(accuracies.avg) + '\n'
    f.write(line)
    f.close()
