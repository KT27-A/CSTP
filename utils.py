import csv
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header, overlay=True):
        if overlay:
            self.log_file = open(path, 'w')
            self.logger = csv.writer(self.log_file, delimiter='\t')
            self.logger.writerow(header)
        else:
            self.log_file = open(path, 'a')
            self.logger = csv.writer(self.log_file, delimiter='\t')
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))

    n_correct_elems = correct.float().sum().item()
    return n_correct_elems / batch_size


def calculate_accuracy5(outputs, targets):
    batch_size = targets.size(0)

    pred = outputs.topk(5, 1, True)[1].data[0].tolist()
    print("true = ", targets.view(1, -1).data[0].tolist()[0], "pred = ", pred)
    correct = targets.view(1, -1).data[0].tolist()[0] in pred
    print(correct)
    n_correct_elems = int(correct)

    return n_correct_elems / batch_size


def calculate_accuracy_video(output_buffer, i):
    true_value = output_buffer[:i+1, -1]
    pred_value = np.argmax(output_buffer[:i+1, :-1], axis=1)
#    print(output_buffer[0:3,:])
    # print(true_value)
    # print(pred_value)
    # print("accuracy = ", 1*(np.equal(true_value, pred_value)).sum()/len(true_value))
    return 1 * (np.equal(true_value, pred_value)).sum()/len(true_value)


def get_dataloader(dataset, opts, data_type='train'):
    if opts.distributed:
        if data_type == "byol":
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                      num_replicas=opts.world_size,
                                                                      rank=opts.rank,
                                                                      shuffle=True)
            opts.batch_size = int(opts.batch_size / opts.world_size)
            data_loader = DataLoader(dataset,
                                     batch_size=opts.batch_size,
                                     shuffle=False,
                                     num_workers=opts.n_workers,
                                     pin_memory=True,
                                     sampler=sampler,
                                     drop_last=True)
        elif data_type == "train":
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                      num_replicas=opts.world_size,
                                                                      rank=opts.rank,
                                                                      shuffle=True)
            batch_size = int(opts.batch_size / opts.world_size)
            data_loader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=opts.n_workers,
                                     pin_memory=True,
                                     sampler=sampler,
                                     drop_last=True)

        elif data_type == "val":
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                      num_replicas=opts.world_size,
                                                                      rank=opts.rank,
                                                                      shuffle=False)
            batch_size = int(opts.batch_size / opts.world_size)
            data_loader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=opts.n_workers,
                                     pin_memory=True,
                                     sampler=sampler,
                                     drop_last=False)

        return data_loader, sampler
    else:
        if data_type == 'train':
            data_loader = DataLoader(dataset,
                                     batch_size=opts.batch_size,
                                     shuffle=True,
                                     num_workers=opts.n_workers,
                                     pin_memory=True,
                                     drop_last=True)
        elif data_type == 'val' or data_type == 'test':
            data_loader = DataLoader(dataset,
                                     batch_size=opts.batch_size,
                                     shuffle=False,
                                     num_workers=opts.n_workers,
                                     pin_memory=True,
                                     drop_last=False)
        return data_loader


# from: https://github.com/pytorch/pytorch/issues/15849#issuecomment-518126031
class _RepeatSampler(object):
    """ Sampler that repeats forever.
    opts:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


# https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
class FastDataLoader(DataLoader):
    '''for reusing cpu workers, to save time'''
    def __init__(self, *opts, **kwopts):
        super().__init__(*opts, **kwopts)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
