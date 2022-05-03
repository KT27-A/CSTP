from data_process.datasets import UcfRepreBYOLSpPre, UCF101RepreLMDB, Kin400RepreLMDB
from data_process.preprocess_data import get_transforms
import os
import torch
from torch import nn
from torch import optim
from models.model import generate_model
from opts import parse_opts
from loss import NTXent
from utils import Logger
import numpy as np
import random
import builtins
from utils import get_dataloader
from utils import AverageMeter
import time
import torch.distributed as dist
from scheduler.cosine_anneal import CosineAnnealingWarmupRestarts


def train_BYOL(epoch, train_dataloader, model, criterion, optimizer, opts, train_logger):
    def reduce_mean(tensor, world_size):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= world_size
        return rt

    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_total_meter = AverageMeter()
    end_time = time.time()
    criterion_cls = criterion[0]
    loss_byol_meter = AverageMeter()
    loss_pred_spa_meter = AverageMeter()
    loss_pred_tem_meter = AverageMeter()
    loss_pred_pb_meter = AverageMeter()
    loss_pred_rot_meter = AverageMeter()

    # contrastive learning
    for i, (inputs, targets) in enumerate(train_dataloader):
        data_time.update(time.time() - end_time)
        clip_1 = inputs[0]
        clip_2 = inputs[1]
        spa_label = targets[0]
        tem_label = targets[1]
        pb_label = targets[2]
        rot_label_1 = targets[3][0]
        rot_label_2 = targets[3][1]
        
        if opts.cuda:
            clip_1 = clip_1.cuda(opts.local_rank, non_blocking=True)
            clip_2 = clip_2.cuda(opts.local_rank, non_blocking=True)
            spa_label = spa_label.cuda(opts.local_rank, non_blocking=True)
            tem_label = tem_label.cuda(opts.local_rank, non_blocking=True)
            pb_label = pb_label.cuda(opts.local_rank, non_blocking=True)
            rot_label_1 = rot_label_1.cuda(opts.local_rank, non_blocking=True)
            rot_label_2 = rot_label_2.cuda(opts.local_rank, non_blocking=True)

        loss_byol, (pred_spa, pred_tem, pred_pb_1, pred_pb_2, pred_rot_1, pred_rot_2) = model(clip_1, clip_2, o_type=opts.task)

        loss_byol = loss_byol.mean()
        loss_pred_spa = criterion_cls(pred_spa, spa_label)
        loss_pred_tem = criterion_cls(pred_tem, tem_label)
        loss_pred_pb_1 = criterion_cls(pred_pb_1, pb_label)
        loss_pred_pb_2 = criterion_cls(pred_pb_2, pb_label)
        loss_pred_rot_1 = criterion_cls(pred_rot_1, rot_label_1)
        loss_pred_rot_2 = criterion_cls(pred_rot_2, rot_label_2)
        # dist.barrier()
        loss_weight = opts.loss_weight
        loss_total = loss_weight[0] * loss_byol + loss_weight[1] * loss_pred_spa + loss_weight[2] * loss_pred_tem + \
            loss_weight[3] * loss_pred_pb_1 + loss_weight[3] * loss_pred_pb_2 + loss_weight[4] * loss_pred_rot_1 + \
            loss_weight[4] * loss_pred_rot_2

        reduced_loss = reduce_mean(loss_total, opts.world_size)
        clip_size = clip_1.size(0)
        loss_total_meter.update(reduced_loss.item(), clip_size)
        loss_byol_meter.update(loss_byol.item(), clip_size)
        loss_pred_spa_meter.update(loss_pred_spa.item(), clip_size)
        loss_pred_tem_meter.update(loss_pred_tem.item(), clip_size)
        loss_pred_pb = (loss_pred_pb_1 + loss_pred_pb_2) / 2
        loss_pred_pb_meter.update(loss_pred_pb.item(), clip_size)
        loss_pred_rot = (loss_pred_rot_1 + loss_pred_rot_2) / 2
        loss_pred_rot_meter.update(loss_pred_rot.item(), clip_size)
        
        optimizer.zero_grad()
        loss_total.backward()
        if opts.clip_grad_norm:
            clip_value = 18
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print("Epoch: [{0}][{1}/{2}]\t"
              "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
              "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
              "Loss_byol {loss_byol.val:.4f} ({loss_byol.avg:.4f})\t"
              "Loss_pred_spa {loss_pred_spa.val:.4f} ({loss_pred_spa.avg:.4f})\t"
              "Loss_pred_tem {loss_pred_tem.val:.4f} ({loss_pred_tem.avg:.4f})\t"
              "Loss_pred_pb {loss_pred_pb.val:.4f} ({loss_pred_pb.avg:.4f})\t"
              "Loss_pred_rot {loss_pred_rot.val:4f} ({loss_pred_rot.avg:.4f})"
              "Loss_total {loss_total.val:.4f} ({loss_total.avg:.4f})\t"
              "Lr {lr:.4}".format(
                        epoch,
                        i + 1,
                        len(train_dataloader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss_byol=loss_byol_meter,
                        loss_pred_spa=loss_pred_spa_meter,
                        loss_pred_tem=loss_pred_tem_meter,
                        loss_pred_pb=loss_pred_pb_meter,
                        loss_pred_rot=loss_pred_rot_meter,
                        loss_total=loss_total_meter,
                        lr=optimizer.param_groups[-1]['lr']))

    if opts.local_rank == 0:
        train_logger.log({
            "epoch": epoch,
            "loss": loss_total_meter.avg,
            "loss_byol": loss_byol_meter.avg,
            "loss_pred_spa": loss_pred_spa_meter.avg,
            "loss_pred_tem": loss_pred_tem_meter.avg,
            "loss_pred_pb": loss_pred_pb_meter.avg,
            "loss_pred_rot": loss_pred_rot_meter.avg,
            "acc": None,
            "lr": float('{:.5f}'.format(optimizer.param_groups[-1]['lr']))
        })

        if opts.rank == 0 and epoch % 100 == 0:
            save_file_path = os.path.join(opts.result_path, opts.dataset, opts.task, 'save_{}.pth'.format(epoch))
            states = {
                "epoch": epoch + 1,
                "arch": opts.arch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                }
            torch.save(states, save_file_path)


def main(opts):
    torch.manual_seed(opts.manual_seed)
    np.random.seed(opts.manual_seed)
    random.seed(opts.manual_seed)
    if torch.cuda.is_available():
        if opts.local_rank != -1:
            opts.cuda = True
            opts.world_size = int(os.environ["WORLD_SIZE"])
            opts.distributed = True
            opts.nprocs = torch.cuda.device_count()
            main_worker(opts.local_rank, opts.nprocs, opts)
        else:
            opts.distributed = False
            opts.cuda = True
            main_worker('cuda:0', opts.nprocs, opts)
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")


def main_worker(local_rank, ngpus_per_node, opts):
    opts.device = local_rank if opts.cuda else 'cpu'
    if opts.distributed:
        # suppress printing if not master
        if local_rank != 0:
            def print_pass(*opts):
                pass
            builtins.print = print_pass
        opts.rank = local_rank
        dist.init_process_group(backend=opts.dist_backend,
                                init_method=opts.dist_url,
                                world_size=opts.world_size,
                                rank=opts.rank)

        # build log
        log_path = os.path.join(opts.result_path, opts.dataset, opts.task)
        if not os.path.exists(log_path) and local_rank == 0:
            os.makedirs(log_path)
    else:
        log_path = os.path.join(opts.result_path, opts.dataset, opts.task)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    # print opts
    print(opts)
    opts.arch = '{}-{}'.format(opts.model_name, opts.model_depth)

    # define loss
    criterion_cls = nn.CrossEntropyLoss().cuda(opts.device)
    criterion_ctr = NTXent.NTXentLoss(
        device=opts.device,
        batch_size=opts.batch_size,
        temperature=opts.temperature,
        use_cosine_similarity=True
    ).cuda(opts.device)
    criterion = [criterion_cls, criterion_ctr]

    # define transforms and dataloader
    train_transform = get_transforms(mode='pre_train', opts=opts)
    print("Preprocessing train data ...")
    train_data = globals()['{}'.format(opts.dataset)](data_type='train',
                                                      opts=opts,
                                                      split=opts.split,
                                                      sp_transform=train_transform)
    print("Length of training data = ", len(train_data))
    train_dataloader, train_sampler = get_dataloader(train_data, opts=opts, data_type='train')

    # Load the model
    print("Loading model... ", opts.model_name, opts.model_depth)
    model, parameters = generate_model(opts)
    print("Model is loaded successfully!")

    if opts.task == 'resume':
        begin_epoch = int(opts.resume_md_path.split('/')[-1].split('_')[1])
        train_logger = Logger(os.path.join(log_path, '{}_train_clip{}model{}{}.log'.format(
                        opts.dataset, opts.sample_duration, opts.model_name, opts.model_depth)),
                        ["epoch", "loss", "loss_byol", "loss_pred_spa", "loss_pred_tem", "loss_pred_pb", "loss_pred_rot", "acc", "lr"],
                        overlay=False)
    else:
        begin_epoch = 1
        train_logger = Logger(os.path.join(log_path, '{}_train_clip{}model{}{}.log'.format(
                        opts.dataset, opts.sample_duration, opts.model_name, opts.model_depth)),
                        ["epoch", "loss", "loss_byol", "loss_pred_spa", "loss_pred_tem", "loss_pred_pb", "loss_pred_rot", "acc", "lr"],
                        overlay=True)

    # build optimizer
    if opts.optimizer == 'sgd':
        optimizer = optim.SGD(parameters,
                              lr=opts.learning_rate,
                              momentum=opts.momentum,
                              weight_decay=opts.weight_decay)
    elif opts.optimizer == 'adamw':
        optimizer = optim.AdamW(parameters,
                                lr=opts.learning_rate,
                                betas=(0.9, 0.99),
                                weight_decay=opts.weight_decay)
    elif opts.optimizer == 'adam':
        optimizer = optim.Adam(parameters,
                               lr=opts.learning_rate,
                               weight_decay=opts.weight_decay)

    if opts.task == 'resume':
        optimizer.load_state_dict(torch.load(opts.resume_md_path)['optimizer'])

    # build learning rate strategy
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opts.lr_patience)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opts.n_epochs, eta_min=opts.lr_decay*opts.learning_rate)
    # scheduler = CosineAnnealingWarmupRestarts(
    # optimizer, first_cycle_steps=300, cycle_mult=1.0, max_lr=0.03, min_lr=0.00001, warmup_steps=10, gamma=0.5)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=opts.n_epochs,
                                              cycle_mult=1.0,
                                              max_lr=opts.learning_rate,
                                              min_lr=0.00001,
                                              warmup_steps=0.5*opts.n_epochs,
                                              gamma=0.5)

    torch.backends.cudnn.benchmark = True
    # Training and Validation
    if opts.task in ['r_byol', "loss_com"]:
        print('Start to train BYOL CoCLR data augmentation pre-trained model!')
        for epoch in range(begin_epoch, opts.n_epochs + 1):
            print('Training BYOL at epoch {}'.format(epoch))
            if opts.distributed:
                train_sampler.set_epoch(epoch)
            train_BYOL(epoch, train_dataloader, model, criterion, optimizer, opts, train_logger)
            scheduler.step()


if __name__ == "__main__":
    opts = parse_opts()
    main(opts)
