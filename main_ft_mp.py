from data_process.datasets import UcfFineTune, UcfFineTuneLMDB, Kin400FTOfflineLMDB
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
from utils import AverageMeter, calculate_accuracy
import time
import torch.distributed as dist
from scheduler.cosine_anneal import CosineAnnealingWarmupRestarts


def reduce_mean(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


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
        print("-------No CUDA Training!--------")
        opts.distributed = False
        opts.cuda = False
        main_worker("cpu", opts.nprocs, opts)


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

    # define transforms and dataloader
    if opts.task in ['ft_fc', 'ft_all', 'scratch', 'resume']:
        # Tran transform
        train_transform = get_transforms(mode=opts.transform_mode, opts=opts)
        print("Preprocessing train data ...")
        train_data = globals()['{}'.format(opts.dataset)](data_type='train',
                                                          opts=opts,
                                                          split=opts.split,
                                                          sp_transform=train_transform)
        len_train_data = len(train_data)
        print("Length of training data = ", len_train_data)

        # Train dataloader
        train_dataloader, train_sampler = get_dataloader(train_data, opts=opts, data_type='train')

        # Val transform
        val_transform = get_transforms(mode='{}_val'.format(opts.transform_mode), opts=opts)
        print("Preprocessing validation data ...")

        # Val dataloader
        val_data = globals()['{}'.format(opts.dataset)](data_type='val',
                                                        opts=opts,
                                                        split=opts.split,
                                                        sp_transform=val_transform)
        len_val_data = len(val_data)
        print("Length of validation data = ", len_val_data)
        val_dataloader, val_sampler = get_dataloader(val_data, opts=opts, data_type='val')

    # Load the model
    print("Loading model... ", opts.model_name, opts.model_depth)
    model, parameters = generate_model(opts)

    criterion_cls = nn.CrossEntropyLoss().cuda(opts.device)
    criterion_ctr = NTXent.NTXentLoss(device=opts.device,
                                      batch_size=opts.batch_size,
                                      temperature=opts.temperature,
                                      use_cosine_similarity=True).cuda(opts.device)
    criterion = [criterion_cls, criterion_ctr]

    if opts.task == 'resume':
        begin_epoch = int(opts.resume_md_path.split('/')[-1].split('_')[1])
        train_logger = Logger(os.path.join(log_path, '{}_train_clip{}model{}{}.log'.format(
                              opts.dataset, opts.sample_duration, opts.model_name, opts.model_depth)),
                              ['epoch', 'loss', 'acc', 'lr'], overlay=False)
        val_logger = Logger(os.path.join(log_path, '{}_val_clip{}model{}{}.log'.format(
                            opts.dataset, opts.sample_duration, opts.model_name, opts.model_depth)),
                            ['epoch', 'loss', 'acc'], overlay=False)
    else:
        begin_epoch = 1
        train_logger = Logger(os.path.join(log_path, '{}_train_clip{}model{}{}.log'.format(
                              opts.dataset, opts.sample_duration, opts.model_name, opts.model_depth)),
                              ['epoch', 'loss', 'acc', 'lr'], overlay=True)
        val_logger = Logger(os.path.join(log_path, '{}_val_clip{}model{}{}.log'.format(
                            opts.dataset, opts.sample_duration, opts.model_name, opts.model_depth)),
                            ['epoch', 'loss', 'acc'], overlay=True)
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opts.lr_patience)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opts.n_epochs, eta_min=opts.lr_decay*opts.learning_rate)
    # scheduler = CosineAnnealingWarmupRestarts(
    #    optimizer, first_cycle_steps=300, cycle_mult=1.0, max_lr=0.03, min_lr=0.00001, warmup_steps=10, gamma=0.5)
    # scheduler = CosineAnnealingWarmupRestarts(optimizer,
    #                                           first_cycle_steps=opts.n_epochs,
    #                                           cycle_mult=1.0,
    #                                           max_lr=opts.learning_rate,
    #                                           min_lr=0.00001,
    #                                           warmup_steps=0.1*opts.n_epochs,
    #                                           gamma=0.5)

    torch.backends.cudnn.benchmark = True
    # Training and Validation
    if opts.task in ['ft_fc', 'ft_all', 'scratch', 'resume']:
        for epoch in range(begin_epoch, opts.n_epochs + 1):
            print('Start to fine-tune')
            print('Start training epoch {}'.format(epoch))
            if opts.distributed:
                train_sampler.set_epoch(epoch)
            train(epoch, train_dataloader, model, criterion, optimizer, opts, train_logger, len_train_data)
            print('Start validating epoch {}'.format(epoch))
            validation(epoch, val_dataloader, model, criterion, optimizer, opts, val_logger, len_val_data, scheduler)
            # scheduler.step()


def train(epoch, train_dataloader, model, criterion, optimizer, opts, train_logger, len_train_data):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    end_time = time.time()

    # CrossEntropy
    criterion_cls = criterion[0]
    train_prefetcher = data_prefetcher(train_dataloader, opts)
    inputs, targets = train_prefetcher.next()
    i = 0
    while inputs is not None:
        i += 1
        data_time.update(time.time() - end_time)
        assert opts.task in ['scratch', 'r_cls', 'ft_fc', 'ft_all']
        # inputs = inputs.to(opts.device, non_blocking=True)
        # targets = inputs.to(opts.device, non_blocking=True)
        if opts.task in ['']:
            outputs = model(inputs, None, 'cls_nofc')
        elif opts.task in ['ft_fc', 'ft_all', 'r_cls', 'scratch']:
            outputs = model(inputs, o_type=opts.task)

        loss = criterion_cls(outputs, targets)
        acc = calculate_accuracy(outputs, targets)
        # dist.barrier()
        reduced_loss = reduce_mean(loss, opts.world_size)
        # reduced_acc = reduce_mean(acc, opts.world_size)
        losses.update(reduced_loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
              'Lr {lr:.6f}\t'
              "Left {left:.1f}d".format(
                    epoch,
                    i,
                    int(len_train_data / opts.batch_size),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    acc=accuracies,
                    lr=optimizer.param_groups[-1]['lr'],
                    left=(batch_time.avg * ((opts.n_epochs - epoch) * int(len_train_data / opts.batch_size) +
                          int(len_train_data / opts.batch_size) - i)) / 3600 / 24))

        inputs, targets = train_prefetcher.next()

    if opts.rank == 0 and opts.local_rank == 0:
        train_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': float('{:.5f}'.format(optimizer.param_groups[-1]['lr']))
        })


def validation(epoch, val_dataloader, model, criterion, optimizer, opts, val_logger, len_val_data, scheduler):
    def step(inputs, targets, i):
        end_time = time.time()
        data_time.update(time.time() - end_time)
        assert opts.task in ['scratch', 'r_cls', 'ft_fc', 'ft_all']
        if opts.task in ['']:
            outputs = (inputs, None, 'cls_nofc')
        elif opts.task in ['ft_fc', 'ft_all', 'r_cls', 'scratch']:
            outputs = model(inputs, o_type=opts.task)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Val_Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch,
                                                         i + 1,
                                                         int(len_val_data / opts.batch_size),
                                                         batch_time=batch_time,
                                                         data_time=data_time,
                                                         loss=losses,
                                                         acc=accuracies))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    criterion = criterion[0]
    # change to evaluation mode
    model.eval()
    with torch.no_grad():
        val_prefetcher = data_prefetcher(val_dataloader, opts)
        inputs, targets = val_prefetcher.next()
        i = 0
        while inputs is not None:
            step(inputs, targets, i)
            i += 1
            inputs, targets = val_prefetcher.next()

        if opts.rank == 0 and opts.local_rank == 0:
            scheduler.step(losses.avg)
            val_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
            accuracy_val = accuracies.avg
            if accuracy_val > list(opts.highest_val.values())[0]:
                old_key = list(opts.highest_val.keys())[0]
                file_path = os.path.join(opts.result_path, opts.dataset, opts.task, old_key)
                if os.path.exists(file_path):
                    os.remove(file_path)
                opts.highest_val.pop(old_key)
                opts.highest_val['save_{}_max.pth'.format(epoch)] = accuracy_val
                save_file_path = os.path.join(opts.result_path,
                                              opts.dataset,
                                              opts.task,
                                              'save_{}_max.pth'.format(epoch))
                states = {'epoch': epoch + 1,
                          'arch': opts.arch,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
                torch.save(states, save_file_path)


class data_prefetcher():
    def __init__(self, loader, opts):
        self.loader = iter(loader)
        self.opts = opts
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.to(self.opts.device, non_blocking=True)
            self.next_target = self.next_target.to(self.opts.device, non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


if __name__ == "__main__":
    opts = parse_opts()
    main(opts)
