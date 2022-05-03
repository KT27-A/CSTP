from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import glob
import random
import torch
import decord
from torchvision import transforms
import cv2
import lmdb
import msgpack
from io import BytesIO


PACE = [1, 2, 4, 8]
OVERLAP_TEM_RATE = [1.0, 0.8, 0.6, 0.4, 0.2]
ROTATE = [0, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]


class UcfBYOLOnline(Dataset):
    """Online training UCF101 Dataset"""
    def __init__(self, data_type, opts, split, sp_transform):
        """
        Args:
            opts   : config options
            train : train for training, val for validation, test for testing
            split : 1, 2, 3
        Returns:
            list: list of PIL images
        """
        self.data_type = data_type
        self.opts = opts
        self.sp_transform = sp_transform
        self.toPIL = transforms.ToPILImage()  # convert tensor CHW or np HWC -> PIL
        # indexes for training/test set
        if self.data_type == 'train':
            split_name = 'trainlist0{}.txt'.format(split)
        else:
            split_name = 'testlist0{}.txt'.format(split)
        self.data = []  # (filename, id)
        with open(os.path.join(self.opts.annotation_path, split_name), 'r') as f:
            for line in f:
                line_split = line.strip().split(" ")
                video_path = os.path.join(self.opts.frame_dir, line_split[0])
                if os.path.exists(video_path):
                    self.data.append((video_path, line_split[1]))
                else:
                    print('{} does not exist'.format(video_path))

    def __len__(self):
        '''
        returns number of test set
        '''
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        video_path = video[0]
        assert self.opts.task in ['r_byol']
        if self.opts.task == 'r_byol':
            clip_1 = self.repre_train_clip(self.opts, video_path)
            clip_2 = self.repre_train_clip(self.opts, video_path)
            clip_cat = clip_1 + clip_2
            clip_1, clip_2 = self.sp_transform(clip_cat)
            return torch.stack(clip_1).transpose(0, 1), torch.stack(clip_2).transpose(0, 1)

    def repre_train_clip(self, opts, frame_path):
        """
        Chooses a random clip from a video for training/ validation
        Args:
            opts         : config options
            frame_path  : frames of video frames
            total_frames: Number of frames in the video
        Returns:
            list(frames) : random clip
        """
        vr = decord.VideoReader(frame_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        clip = []
        label = 0  # skip no frame
        sample_rate = PACE[label]

        # choosing a random set of frame
        if total_frames - opts.sample_duration * sample_rate < 0:
            sequence = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip = np.zeros_like(sequence)
            sequence = sequence[sequence < total_frames]
            index_clip[-len(sequence):] = sequence
        else:
            start_frame = np.random.randint(0, total_frames - opts.sample_duration * sample_rate)
            index_clip = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip = index_clip + start_frame
        # print("{}: start{} total{} index{}".format(frame_path, start_frame, total_frames, index_clip))
        clip_numpy = vr.get_batch(index_clip).asnumpy()
        for frame in clip_numpy:
            img = self.toPIL(frame)
            clip.append(img)
        return clip


class UcfBYOLOnlineSelfTrans(Dataset):
    """Online training UCF101 Dataset"""
    def __init__(self, data_type, opts, split, sp_transform):
        """
        Args:
            opts   : config options
            train : train for training, val for validation, test for testing
            split : 1, 2, 3
        Returns:
            list: list of PIL images
        """
        self.data_type = data_type
        self.opts = opts
        self.sp_transform = sp_transform
        self.toPIL = transforms.ToPILImage()  # convert tensor CHW or np HWC -> PIL
        # indexes for training/test set
        if self.data_type == 'train':
            split_name = 'trainlist0{}.txt'.format(split)
        else:
            split_name = 'testlist0{}.txt'.format(split)
        self.data = []  # (filename, id)
        with open(os.path.join(self.opts.annotation_path, split_name), 'r') as f:
            for line in f:
                line_split = line.strip().split(" ")
                video_path = os.path.join(self.opts.frame_dir, line_split[0])
                if os.path.exists(video_path):
                    self.data.append((video_path, line_split[1]))
                else:
                    print('{} does not exist'.format(video_path))

    def __len__(self):
        '''
        returns number of test set
        '''
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        video_path = video[0]
        assert self.opts.task in ['r_byol']
        if self.opts.task == 'r_byol':
            clip_1 = self.repre_train_clip(self.opts, video_path)
            clip_2 = self.repre_train_clip(self.opts, video_path)
            clip_cat = clip_1 + clip_2
            clip_1, clip_2 = self.sp_transform(clip_cat)
            return torch.stack(clip_1).transpose(0, 1), torch.stack(clip_2).transpose(0, 1)

    def repre_train_clip(self, opts, frame_path):
        """
        Chooses a random clip from a video for training/ validation
        Args:
            opts         : config options
            frame_path  : frames of video frames
            total_frames: Number of frames in the video
        Returns:
            list(frames) : random clip
        """
        vr = decord.VideoReader(frame_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        clip = []
        label = 0  # skip no frame
        sample_rate = PACE[label]

        # choosing a random set of frame
        if total_frames - opts.sample_duration * sample_rate < 0:
            sequence = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip = np.zeros_like(sequence)
            sequence = sequence[sequence < total_frames]
            index_clip[-len(sequence):] = sequence
        else:
            start_frame = np.random.randint(0, total_frames - opts.sample_duration * sample_rate)
            index_clip = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip = index_clip + start_frame
        # print("{}: start{} total{} index{}".format(frame_path, start_frame, total_frames, index_clip))
        clip_numpy = vr.get_batch(index_clip).asnumpy()
        for frame in clip_numpy:
            img = self.toPIL(frame)
            clip.append(img)
        return clip


class UCFFTOnline(Dataset):
    """UCF101 Dataset"""
    def __init__(self, data_type, opts, split, sp_transform):
        """
        Args:
            opts   : config options
            train : train for training, val for validation, test for testing
            split : 1,2,3
        Returns:
            (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.data_type = data_type
        self.opts = opts
        self.sp_transform = sp_transform
        self.toPIL = transforms.ToPILImage()  # convert tensor CHW or np HWC -> PIL

        # Build class index
        with open(os.path.join(self.opts.annotation_path, "classInd.txt")) as lab_file:
            self.lab_names = [line.strip('\n').split(' ')[1] for line in lab_file]

        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 101

        # indexes for training/test set
        if self.data_type == 'train':
            split_name = 'trainlist0{}.txt'.format(split)
        else:
            split_name = 'testlist0{}.txt'.format(split)

        self.data = []  # (filename, id, total_frames)
        with open(os.path.join(self.opts.annotation_path, split_name), 'r') as f:
            for line in f:
                line_split = line.strip().split(" ")
                video_path = os.path.join(self.opts.frame_dir, line_split[0])
                if os.path.exists(video_path):
                    self.data.append((video_path, int(line_split[1])))
                else:
                    print('{} does not exist'.format(video_path))

    def __len__(self):
        '''
        returns number of test set
        '''
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        video_path = video[0]
        label_id = video[1]

        if self.data_type == 'test':
            clip_batch = get_test_clip_coclr(self.opts, video_path, self.sp_transform)
            return np.stack(clip_batch).transpose(0, 2, 1, 3, 4), label_id
            # clip = get_test_clip_tf(self.opts, video_path, total_frames, self.sp_transform)
        elif self.data_type == 'val':
            clip = self.get_val_clip(self.opts, video_path)
            clip = self.sp_transform(clip)
        elif self.data_type == 'train':
            clip = self.get_train_clip(self.opts, video_path)
            clip = self.sp_transform(clip)

        return torch.stack(clip).transpose(0, 1), label_id

    def get_train_clip(self, opts, frame_path):
        sample_rate = 1
        clip = []
        vr = decord.VideoReader(frame_path)
        total_frames = len(vr)

        if total_frames - opts.sample_duration * sample_rate < 0:
            sequence = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip = np.zeros_like(sequence)
            sequence = sequence[sequence < total_frames]
            index_clip[-len(sequence):] = sequence
            start_frame = 1
        else:
            start_frame = np.random.randint(0, total_frames - opts.sample_duration * sample_rate)
            index_clip = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip += start_frame

        clip_numpy = vr.get_batch(index_clip).asnumpy()
        for frame in clip_numpy:
            img = self.toPIL(frame)
            clip.append(img)
        return clip

    def get_val_clip(self, opts, frame_path):
        sample_rate = 1
        clip = []
        vr = decord.VideoReader(frame_path)
        total_frames = len(vr)

        if total_frames - opts.sample_duration * sample_rate < 0:
            sequence = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip = np.zeros_like(sequence)
            sequence = sequence[sequence < total_frames]
            index_clip[-len(sequence):] = sequence
            start_frame = 1
        else:
            start_frame = np.random.randint(0, total_frames - opts.sample_duration * sample_rate)
            index_clip = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip += start_frame

        clip_numpy = vr.get_batch(index_clip).asnumpy()
        for frame in clip_numpy:
            img = self.toPIL(frame)
            clip.append(img)
        return clip


class UcfRepre(Dataset):
    """UCF101 Dataset"""
    def __init__(self, data_type, opts, split, sp_transform):
        """
        Args:
            opts   : config options
            train : train for training, val for validation, test for testing
            split : 1,2,3 
        Returns:
            (tensor(frames), class_id, total_frames): Shape of tensor C x T x H x W
        """
        self.data_type = data_type
        self.opts = opts
        self.sp_transform = sp_transform
        # indexes for training/test set
        if self.data_type == 'train':
            split_name = 'trainlist0{}_nframe.txt'.format(split)
        else:
            split_name = 'testlist0{}_nframe.txt'.format(split)
        self.data = []  # (filename, id, total_frames)
        with open(os.path.join(self.opts.annotation_path, split_name), 'r') as f:
            for line in f:
                line_split = line.strip().split(" ")
                video_path = os.path.join(self.opts.frame_dir, line_split[0].split('.')[0])
                if os.path.exists(video_path):
                    total_frames = line_split[2]
                    self.data.append((video_path, line_split[1], int(total_frames)))
                else:
                    print('{} does not exist'.format(video_path))

    def __len__(self):
        '''
        returns number of test set
        '''
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        video_path = video[0]
        total_frames = video[2]
        assert self.opts.task in ['r_cls', 'r_ctr']
        if self.opts.task == 'r_cls':
            clip, pace_label = self.repre_train_clip(self.opts, video_path, total_frames)
            clip = self.clip_process(clip, self.sp_transform, self.opts.sample_size)
            return clip, pace_label
        elif self.opts.task == 'r_ctr':
            clip_1, pace_label_1 = self.repre_train_clip(self.opts, video_path, total_frames)
            clip_2, pace_label_2 = self.repre_train_clip(self.opts, video_path, total_frames)
            clip_1 = self.clip_process(clip_1, self.sp_transform, self.opts.sample_size)
            clip_2 = self.clip_process(clip_2, self.sp_transform, self.opts.sample_size)
            return (clip_1, clip_2), (pace_label_1, pace_label_2)

    def repre_val_clip(self, opts, frame_path, total_frames):
        """
            Chooses a random clip from a video for training/ validation
            Args:
                opts         : config options
                frame_path  : frames of video frames
                total_frames: Number of frames in the video
            Returns:
                list(frames) : random clip
        """
        clip = []
        i = 0
        label = 0
        sample_rate = 1
        if total_frames > opts.sample_duration:
            start_frame = np.random.randint(1, total_frames - opts.sample_duration)
        else:
            start_frame = 1

        # choosing a random frame
        while len(clip) < opts.sample_duration:
            try:
                im = Image.open(os.path.join(frame_path, '%05d.jpg' % (start_frame + (i * sample_rate))))
            except ValueError:
                print("Reading img error of {}".format(
                    os.path.join(frame_path, '%05d.jpg' % (start_frame + (i * sample_rate)))))
                clip.append(im.copy())
                im.close()
            if start_frame + ((i+1) * sample_rate) > total_frames:
                start_frame = 1
                i = 0
            else:
                i += 1
        return clip, label

    def repre_train_clip(self, opts, frame_path, total_frames):
        """
            Chooses a random clip from a video for training/ validation
            Args:
                opts         : config options
                frame_path  : frames of video frames
                total_frames: Number of frames in the video
            Returns:
                list(frames) : random clip
            """
        clip = []
        i = 0

        label = random.randint(0, 3)
        sample_rate = PACE[label]
        start_frame = np.random.randint(1, total_frames)

        # choosing a random frame
        while len(clip) < opts.sample_duration:
            try:
                img = Image.open(os.path.join(frame_path, '%05d.jpg' % (start_frame + (i * sample_rate))))
            except ValueError:
                print("Reading img error of {}".format(
                            os.path.join(frame_path, '%05d.jpg' % (start_frame + (i * sample_rate)))))
            clip.append(img.copy())
            img.close()

            if start_frame + ((i+1) * sample_rate) > total_frames:
                start_frame = 1
                i = 0
            else:
                i += 1

        return clip, label


class UcfRepreBYOL(Dataset):
    """UCF101 Dataset"""
    def __init__(self, data_type, opts, split, sp_transform):
        """
        Args:
            opts   : config options
            train : train for training, val for validation, test for testing
            split : 1,2,3
        Returns:
            (tensor(frames), class_id, total_frames): Shape of tensor C x T x H x W
        """
        self.data_type = data_type
        self.opts = opts
        self.sp_transform = sp_transform
        # indexes for training/test set
        if self.data_type == 'train':
            split_name = 'trainlist0{}_nframe.txt'.format(split)
        else:
            split_name = 'testlist0{}_nframe.txt'.format(split)
        self.data = []  # (filename, id, total_frames)
        with open(os.path.join(self.opts.annotation_path, split_name), 'r') as f:
            for line in f:
                line_split = line.strip().split(" ")
                video_path = os.path.join(self.opts.frame_dir, line_split[0].split('.')[0])
                if os.path.exists(video_path):
                    total_frames = line_split[2]
                    self.data.append((video_path, line_split[1], int(total_frames)))
                else:
                    print('{} does not exist'.format(video_path))

    def __len__(self):
        '''
        returns number of test set
        '''
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        video_path = video[0]
        total_frames = video[2]
        assert self.opts.task in ['r_byol']
        if self.opts.task == 'r_byol':
            clip_1 = self.repre_train_clip(self.opts, video_path, total_frames)
            clip_2 = self.repre_train_clip(self.opts, video_path, total_frames)
            clip_cat = clip_1 + clip_2
            clip_1, clip_2 = self.sp_transform(clip_cat)
            return torch.stack(clip_1).transpose(0, 1), torch.stack(clip_2).transpose(0, 1)

    def repre_train_clip(self, opts, frame_path, total_frames):
        """
            Chooses a random clip from a video for training/ validation
            Args:
                opts         : config options
                frame_path  : frames of video frames
                total_frames: Number of frames in the video
            Returns:
                list(frames) : random clip
        """
        clip = []
        label = random.randint(0, 3)  # temporal random
        sample_rate = PACE[label]

        # choosing a random frame
        if total_frames - opts.sample_duration * sample_rate < 0:
            sequence = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip = np.zeros_like(sequence)
            sequence = sequence[sequence < total_frames]
            index_clip[-len(sequence):] = sequence
            start_frame = 1
        else:
            start_frame = np.random.randint(1, total_frames + 2 - opts.sample_duration * sample_rate)
            index_clip = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
        for i in index_clip:
            try:
                img = Image.open(os.path.join(frame_path, '%05d.jpg' % (start_frame + i)))
            except ValueError:
                print("Reading img error of {}".format(
                                os.path.join(frame_path, '%05d.jpg' % (start_frame + i))))
            clip.append(img.copy())
            img.close()
        return clip


class UCF101RepreLMDB(object):
    def __init__(self, data_type, opts, split, sp_transform):
        self.data_type = data_type
        self.opts = opts
        self.sp_transform = sp_transform

        print('Loading LMDB from %s, split:%s' % (self.opts.lmdb_path, split))
        self.env = lmdb.open(self.opts.lmdb_path, subdir=os.path.isdir(self.opts.lmdb_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            db_order = msgpack.loads(txn.get(b'__order__'))
        
        video_dict = dict(zip([i for i in db_order], ['%09d'%i for i in range(len(db_order))]))

        # indexes for training/test set
        if self.data_type == 'train':
            split_name = 'trainlist0{}_nframe.txt'.format(split)
        else:
            split_name = 'testlist0{}_nframe.txt'.format(split)
        self.data = []  # (video_name, lmdb_key, label, total_frames)
        with open(os.path.join(self.opts.annotation_path, split_name), 'r') as f:
            for line in f:
                split = line.strip().split(" ")
                video_name = split[0].split(".")[0]
                video_label = split[1]
                total_frames = split[2]
                video_key = video_dict[video_name]
                self.data.append((video_name, video_key.encode('ascii'), video_label, int(total_frames)))
        db_order.clear()
        video_dict.clear()

    def pil_from_raw_rgb(self, raw):
        return Image.open(BytesIO(raw)).convert('RGB')

    def __getitem__(self, idx):
        video_name, video_key, _, total_frames = self.data[idx]

        assert self.opts.task in ['r_byol', "loss_com"]
        clip_1, clip_2, tem_label, pb_label, rotate_label = self.repre_train_clip(video_name, video_key, total_frames)
        clip_cat = clip_1 + clip_2
        clip_1, clip_2, spa_label = self.sp_transform(clip_cat)
        return [torch.stack(clip_1).transpose(0, 1), torch.stack(clip_2).transpose(0, 1)], \
               [spa_label, tem_label, pb_label, rotate_label]

    def repre_train_clip(self, video_name, video_key, total_frames):
        """
            Chooses a random clip from a video for training/ validation
            Args:
                frame_path  : frames of video frames
                total_frames: Number of frames in the video
            Returns:
                list(frames) : random clip
        """
        clip_1 = []
        clip_2 = []
        # label = random.randint(0, 3)  # temporal random
        # sample_rate = PACE[label]
        max_pb = int(np.log2(total_frames / (self.opts.sample_duration - 1)))
        pb_label = random.randint(0, min(3, max_pb))
        sample_rate = PACE[pb_label]
        # tem_interval = 0.2
        clip_range = (self.opts.sample_duration - 1) * sample_rate

        rot_label_1 = random.randint(0, 3)
        rotate_angle_1 = ROTATE[rot_label_1]
        rot_label_2 = random.randint(0, 3)
        rotate_angle_2 = ROTATE[rot_label_2]
        # choosing a random frame
        if total_frames - clip_range <= 0:
            index_clip = []
            idx_frame = 0
            while len(index_clip) < self.opts.sample_duration:
                index_clip.append(idx_frame)
                idx_frame += sample_rate
                if idx_frame >= total_frames:
                    idx_frame = 0
            start_frame = 0

            with self.env.begin(write=False) as txn:
                raw = msgpack.loads(txn.get(video_key))

            for i in index_clip:
                try:
                    img = self.pil_from_raw_rgb(raw[start_frame + i])
                    if rotate_angle_1 != 0:
                        img_1 = img.transpose(rotate_angle_1)
                    else:
                        img_1 = img
                    if rotate_angle_2 != 0:
                        img_2 = img.transpose(rotate_angle_2)
                    else:
                        img_2 = img
                except ValueError:
                    print("Reading img error of {}".format(os.path.join(video_name, "%05d.jpg" % (start_frame + i))))
                clip_1.append(img_1.copy())
                clip_2.append(img_2.copy())
                img.close()
            tem_label = 0
            return clip_1, clip_2, tem_label, pb_label, [rot_label_1, rot_label_2]
        else:
            start_frame = random.randint(0, total_frames - clip_range - 1)
            while True:
                tem_label = random.randint(0, 4)
                tem_rate = OVERLAP_TEM_RATE[tem_label]
                front_behind = random.randint(0, 1)  # front of behind of second clip, 0 for front, 1 for behind
                if front_behind == 0:
                    start_frame_2 = start_frame - int((1 - tem_rate) * clip_range)
                    if start_frame_2 < 1:
                        continue
                else:
                    start_frame_2 = start_frame + int((1 - tem_rate) * clip_range)
                    if start_frame_2 > total_frames - clip_range:
                        continue

                index_clip = np.arange(0, clip_range + 1, sample_rate)

                with self.env.begin(write=False) as txn:
                    raw = msgpack.loads(txn.get(video_key))
                for i in index_clip:
                    try:
                        img = self.pil_from_raw_rgb(raw[start_frame + i])
                        if rotate_angle_1 != 0:
                            img = img.transpose(rotate_angle_1)
                    except ValueError:
                        print("Reading img error of {}".format(os.path.join(video_name, "%05d.jpg" % (start_frame + i))))
                    clip_1.append(img.copy())
                    img.close()
                
                with self.env.begin(write=False) as txn:
                    raw = msgpack.loads(txn.get(video_key))

                for i in index_clip:
                    try:
                        img = self.pil_from_raw_rgb(raw[start_frame + i])
                        if rotate_angle_2 != 0:
                            img = img.transpose(rotate_angle_2)
                    except ValueError:
                        print("Reading img error of {}".format(os.path.join(video_name, "%05d.jpg" % (start_frame_2 + i))))
                    clip_2.append(img.copy())
                    img.close()

                return clip_1, clip_2, tem_label, pb_label, [rot_label_1, rot_label_2]

    def __len__(self):
        return len(self.data)


class UcfFineTuneLMDB(Dataset):
    """UCF101 Dataset Finetune"""
    def __init__(self, data_type, opts, split, sp_transform):
        """
        Args:
            opts: config options
            train: train for training, val for validation, test for testing
            split: 1,2,3
        Returns:
            (tensor(frames), class_id, total_frames): Shape of tensor C x T x H x W
        """
        self.data_type = data_type
        self.opts = opts
        self.sp_transform = sp_transform
        # Loading LMDB
        print('Loading LMDB from %s, split:%s' % (self.opts.lmdb_path, split))
        self.env = lmdb.open(self.opts.lmdb_path, subdir=os.path.isdir(self.opts.lmdb_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            db_order = msgpack.loads(txn.get(b'__order__'))
        
        video_dict = dict(zip([i for i in db_order], ['%09d'%i for i in range(len(db_order))]))

        # indexes for training/test set
        if self.data_type == 'train':
            split_name = 'trainlist0{}_nframe.txt'.format(split)
        else:
            split_name = 'testlist0{}_nframe.txt'.format(split)
        self.data = []  # (video_name, lmdb_key, label, total_frames)
        with open(os.path.join(self.opts.annotation_path, split_name), 'r') as f:
            for line in f:
                split = line.strip().split(" ")
                video_name = split[0].split(".")[0]
                video_label = split[1]
                total_frames = split[2]
                video_key = video_dict[video_name]
                self.data.append((video_name, video_key.encode('ascii'), int(video_label), int(total_frames)))
        db_order.clear()
        video_dict.clear()

    def __len__(self):
        '''
        returns number of test set
        '''
        return len(self.data)
    
    def pil_from_raw_rgb(self, raw):
        return Image.open(BytesIO(raw)).convert('RGB')

    def __getitem__(self, idx):
        video_name, video_key, label, total_frames = self.data[idx]
        assert self.opts.task in ["ft_fc", "ft_all", "scratch", "test"]
        if self.data_type == 'train':
            clip = self._get_train_clip(video_name, video_key, total_frames)
            return torch.stack(clip).transpose(0, 1), label
        elif self.data_type == 'val':
            clip = self._get_val_clip(video_name, video_key, total_frames)
            return torch.stack(clip).transpose(0, 1), label
        elif self.data_type == 'test':
            clip_batch = self._get_test_clip(video_name, video_key, total_frames)
            return np.stack(clip_batch).transpose(0, 2, 1, 3, 4), label

    def _get_train_clip(self, video_name, video_key, total_frames):
        sample_rate = self.opts.pb_rate
        clip = []
        clip_range = (self.opts.sample_duration - 1) * sample_rate
        if total_frames - clip_range <= 0:
            index_clip = []
            idx_frame = 0
            while len(index_clip) < self.opts.sample_duration:
                index_clip.append(idx_frame)
                idx_frame += sample_rate
                if idx_frame >= total_frames:
                    idx_frame = 0
            start_frame = 0
        else:
            start_frame = random.randint(0, total_frames - clip_range - 1)
            index_clip = np.arange(0, clip_range + 1, sample_rate)
        
        with self.env.begin(write=False) as txn:
            raw = msgpack.loads(txn.get(video_key))

        for i in index_clip:
            try: img = self.pil_from_raw_rgb(raw[start_frame + i])
            except ValueError:
                print("Reading img error of {}".format(os.path.join(video_name, "%05d.jpg" % (start_frame + i))))
            clip.append(img.copy())
            img.close()

        clip = self.sp_transform(clip)

        return clip

    def _get_val_clip(self, video_name, video_key, total_frames):
        sample_rate = self.opts.pb_rate
        clip = []
        clip_range = (self.opts.sample_duration - 1) * sample_rate
        if total_frames - clip_range <= 0:
            index_clip = []
            idx_frame = 0
            while len(index_clip) < self.opts.sample_duration:
                index_clip.append(idx_frame)
                idx_frame += sample_rate
                if idx_frame >= total_frames:
                    idx_frame = 0
            start_frame = 0
        else:
            start_frame = random.randint(0, total_frames - clip_range - 1)
            index_clip = np.arange(0, clip_range + 1, sample_rate)
        with self.env.begin(write=False) as txn:
            raw = msgpack.loads(txn.get(video_key))
        for i in index_clip:
            try: img = self.pil_from_raw_rgb(raw[start_frame + i])
            except ValueError:
                print("Reading img error of {}".format(os.path.join(video_name, "%05d.jpg" % (start_frame + i))))
            clip.append(img.copy())
            img.close()

        clip = self.sp_transform(clip)

        return clip

    def _get_test_clip(self, video_name, video_key, total_frames):
        clip_batch = []
        sample_rate = self.opts.pb_rate  # can be changed
        clip_range = (self.opts.sample_duration - 1) * sample_rate
        if total_frames - clip_range <= 0:  # pad left, only sample once
            seq_idx = []
            idx_frame = 1
            while len(seq_idx) < self.opts.sample_duration:
                seq_idx.append(idx_frame)
                idx_frame += sample_rate
                if idx_frame >= total_frames:
                    idx_frame = 1
            seq_idx = np.expand_dims(seq_idx, 0)
        else:
            # overlap 0.5
            # start = np.expand_dims(np.arange(0, available+1, self.opts.sample_duration * sample_rate // 2 - 1), 1)
            start = np.expand_dims(np.arange(0, total_frames - clip_range, clip_range), 1)
            seq_idx = np.expand_dims(np.arange(self.opts.sample_duration) * sample_rate, 0) + start
            last = np.expand_dims(np.arange(total_frames - clip_range - 1, total_frames, sample_rate), 0)
            seq_idx = np.append(seq_idx, last, 0)
        
        with self.env.begin(write=False) as txn:
            raw = msgpack.loads(txn.get(video_key))

        for idx in seq_idx:
            clip = []
            for i in idx:
                try: img = self.pil_from_raw_rgb(raw[i])
                except ValueError:
                    print("Reading img error of {}".format(os.path.join(video_name, "%05d.jpg" % i)))
                clip.append(img.copy())
                img.close()

            clip = self.sp_transform(clip)
            clip = torch.stack(clip)
            clip_batch.append(clip)

        return clip_batch


class UcfRepreBYOLSpPre(Dataset):
    """UCF101 Dataset BYOL + speed prediction"""
    def __init__(self, data_type, opts, split, sp_transform):
        """
        Args:
            opts   : config options
            train : train for training, val for validation, test for testing
            split : 1,2,3
        Returns:
            (tensor(frames), class_id, total_frames): Shape of tensor C x T x H x W
        """
        self.data_type = data_type
        self.opts = opts
        self.sp_transform = sp_transform
        # indexes for training/test set
        if self.data_type == 'train':
            split_name = 'trainlist0{}_nframe.txt'.format(split)
        else:
            split_name = 'testlist0{}_nframe.txt'.format(split)
        self.data = []  # (filename, id, total_frames)
        with open(os.path.join(self.opts.annotation_path, split_name), 'r') as f:
            for line in f:
                line_split = line.strip().split(" ")
                video_path = os.path.join(self.opts.frame_dir, line_split[0].split('.')[0])
                if os.path.exists(video_path):
                    total_frames = line_split[2]
                    self.data.append((video_path, line_split[1], int(total_frames)))
                else:
                    print('{} does not exist'.format(video_path))

    def __len__(self):
        '''
        returns number of test set
        '''
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        video_path = video[0]
        total_frames = video[2]
        assert self.opts.task in ['r_byol', "loss_com"]
        clip_1, clip_2, tem_label, pb_label, rotate_label = self.repre_train_clip(video_path, total_frames)
        clip_cat = clip_1 + clip_2
        clip_1, clip_2, spa_label = self.sp_transform(clip_cat)
        return [torch.stack(clip_1).transpose(0, 1), torch.stack(clip_2).transpose(0, 1)], \
               [spa_label, tem_label, pb_label, rotate_label]

    def repre_train_clip(self, frame_path, total_frames):
        """
            Chooses a random clip from a video for training/ validation
            Args:
                frame_path  : frames of video frames
                total_frames: Number of frames in the video
            Returns:
                list(frames) : random clip
        """
        clip_1 = []
        clip_2 = []
        # label = random.randint(0, 3)  # temporal random
        # sample_rate = PACE[label]
        max_pb = int(np.log2(total_frames / (self.opts.sample_duration - 1)))
        pb_label = random.randint(0, min(3, max_pb))
        sample_rate = PACE[pb_label]
        # tem_interval = 0.2
        clip_range = (self.opts.sample_duration - 1) * sample_rate

        rot_label_1 = random.randint(0, 3)
        rotate_angle_1 = ROTATE[rot_label_1]
        rot_label_2 = random.randint(0, 3)
        rotate_angle_2 = ROTATE[rot_label_2]
        # choosing a random frame
        if total_frames - clip_range <= 0:
            index_clip = []
            idx_frame = 0
            while len(index_clip) < self.opts.sample_duration:
                index_clip.append(idx_frame)
                idx_frame += sample_rate
                if idx_frame >= total_frames:
                    idx_frame = 0
            start_frame = 1

            for i in index_clip:
                try:
                    img = Image.open(os.path.join(frame_path, '%05d.jpg' % (start_frame + i)))
                    if rotate_angle_1 != 0:
                        img_1 = img.transpose(rotate_angle_1)
                    else:
                        img_1 = img
                    if rotate_angle_2 != 0:
                        img_2 = img.transpose(rotate_angle_2)
                    else:
                        img_2 = img
                except ValueError:
                    print("Reading img error of {}".format(
                                    os.path.join(frame_path, '%05d.jpg' % (start_frame + i))))
                clip_1.append(img_1.copy())
                clip_2.append(img_2.copy())
                img.close()
            tem_label = 0
            return clip_1, clip_2, tem_label, pb_label, [rot_label_1, rot_label_2]
        else:
            start_frame = random.randint(1, total_frames - clip_range)
            while True:
                tem_label = random.randint(0, 4)
                tem_rate = OVERLAP_TEM_RATE[tem_label]
                front_behind = random.randint(0, 1)  # front of behind of second clip, 0 for front, 1 for behind
                if front_behind == 0:
                    start_frame_2 = start_frame - int((1 - tem_rate) * clip_range)
                    if start_frame_2 < 1:
                        continue
                else:
                    start_frame_2 = start_frame + int((1 - tem_rate) * clip_range)
                    if start_frame_2 > total_frames - clip_range:
                        continue

                index_clip = np.arange(0, clip_range + 1, sample_rate)
                for i in index_clip:
                    try:
                        img = Image.open(os.path.join(frame_path, '%05d.jpg' % (start_frame + i)))
                        if rotate_angle_1 != 0:
                            img = img.transpose(rotate_angle_1)
                    except ValueError:
                        print("Reading img error of {}".format(os.path.join(frame_path, '%05d.jpg' % (start_frame + i))))
                    clip_1.append(img.copy())
                    img.close()

                for i in index_clip:
                    try:
                        img = Image.open(os.path.join(frame_path, "%05d.jpg" % (start_frame_2 + i)))
                        if rotate_angle_2 != 0:
                            img = img.transpose(rotate_angle_2)
                    except ValueError:
                        print("Reading img error of {}".format(os.path.join(frame_path, "%05d.jpg" % (start_frame_2 + i))))
                    clip_2.append(img.copy())
                    img.close()

                return clip_1, clip_2, tem_label, pb_label, [rot_label_1, rot_label_2]


class UcfFineTune(Dataset):
    """UCF101 Dataset BYOL + speed prediction"""
    def __init__(self, data_type, opts, split, sp_transform):
        """
        Args:
            opts   : config options
            train : train for training, val for validation, test for testing
            split : 1,2,3
        Returns:
            (tensor(frames), class_id, total_frames): Shape of tensor C x T x H x W
        """
        self.data_type = data_type
        self.opts = opts
        self.sp_transform = sp_transform
        # indexes for training/test set
        if self.data_type == 'train':
            split_name = 'trainlist0{}_nframe.txt'.format(split)
        else:
            split_name = 'testlist0{}_nframe.txt'.format(split)
        self.data = []  # (filename, id, total_frames)
        with open(os.path.join(self.opts.annotation_path, split_name), 'r') as f:
            for line in f:
                line_split = line.strip().split(" ")
                video_path = os.path.join(self.opts.frame_dir, line_split[0].split('.')[0])
                if os.path.exists(video_path):
                    total_frames = line_split[2]
                    self.data.append((video_path, int(line_split[1]), int(total_frames)))
                else:
                    print('{} does not exist'.format(video_path))

    def __len__(self):
        '''
        returns number of test set
        '''
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        frame_path = video[0]
        label_id = video[1]
        total_frames = video[2]
        assert self.opts.task in ["ft_fc", "ft_all", "scratch", "test"]
        if self.data_type == 'train':
            clip = self._get_train_clip(frame_path, total_frames)
            return torch.stack(clip).transpose(0, 1), label_id
        elif self.data_type == 'val':
            clip = self._get_val_clip(frame_path, total_frames)
            return torch.stack(clip).transpose(0, 1), label_id
        elif self.data_type == 'test':
            clip_batch = self._get_test_clip(frame_path, total_frames, self.sp_transform)
            return np.stack(clip_batch).transpose(0, 2, 1, 3, 4), label_id

    def _get_train_clip(self, frame_path, total_frames):
        sample_rate = self.opts.pb_rate
        clip = []
        clip_range = (self.opts.sample_duration - 1) * sample_rate
        if total_frames - clip_range <= 0:
            index_clip = []
            idx_frame = 0
            while len(index_clip) < self.opts.sample_duration:
                index_clip.append(idx_frame)
                idx_frame += sample_rate
                if idx_frame >= total_frames:
                    idx_frame = 0
            start_frame = 1
        else:
            start_frame = random.randint(1, total_frames - clip_range)
            index_clip = np.arange(0, clip_range + 1, sample_rate)

        for i in index_clip:
            try:
                img = Image.open(os.path.join(frame_path, '%05d.jpg' % (start_frame + i)))
            except ValueError:
                print("Reading img error of {}".format(
                            os.path.join(frame_path, '%05d.jpg' % (start_frame + i))))
            clip.append(img.copy())
            img.close()

        clip = self.sp_transform(clip)

        return clip

    def _get_val_clip(self, frame_path, total_frames):
        sample_rate = self.opts.pb_rate
        clip = []
        clip_range = (self.opts.sample_duration - 1) * sample_rate
        if total_frames - clip_range <= 0:
            index_clip = []
            idx_frame = 0
            while len(index_clip) < self.opts.sample_duration:
                index_clip.append(idx_frame)
                idx_frame += sample_rate
                if idx_frame >= total_frames:
                    idx_frame = 0
            start_frame = 1
        else:
            start_frame = random.randint(1, total_frames - clip_range)
            index_clip = np.arange(0, clip_range + 1, sample_rate)
        for i in index_clip:
            try:
                img = Image.open(os.path.join(frame_path, '%05d.jpg' % (start_frame + i)))
            except ValueError:
                print("Reading img error of {}".format(
                            os.path.join(frame_path, '%05d.jpg' % (start_frame + i))))
            clip.append(img.copy())
            img.close()

        clip = self.sp_transform(clip)

        return clip

    def _get_test_clip(self, frame_path, total_frames, sp_transform):
        clip_batch = []
        sample_rate = self.opts.pb_rate  # can be changed
        clip_range = (self.opts.sample_duration - 1) * sample_rate
        if total_frames - clip_range <= 0:  # pad left, only sample once
            seq_idx = []
            idx_frame = 1
            while len(seq_idx) < self.opts.sample_duration:
                seq_idx.append(idx_frame)
                idx_frame += sample_rate
                if idx_frame >= total_frames:
                    idx_frame = 1
            seq_idx = np.expand_dims(seq_idx, 0)
        else:
            # overlap 0.5
            # start = np.expand_dims(np.arange(0, available+1, self.opts.sample_duration * sample_rate // 2 - 1), 1)
            start = np.expand_dims(np.arange(1, total_frames - clip_range + 1, clip_range), 1)
            seq_idx = np.expand_dims(np.arange(self.opts.sample_duration) * sample_rate, 0) + start
            last = np.expand_dims(np.arange(total_frames - clip_range, total_frames + 1, sample_rate), 0)
            seq_idx = np.append(seq_idx, last, 0)

        for idx in seq_idx:
            clip = []
            for i in idx:
                try:
                    img = Image.open(os.path.join(frame_path, '%05d.jpg' % (i)))  # image start from 1
                except ValueError:
                    print("Reading img error of {}".format(os.path.join(frame_path, '%05d.jpg' % (i))))
                clip.append(img.copy())
                img.close()

            clip = self.sp_transform(clip)
            clip = torch.stack(clip)
            clip_batch.append(clip)

        return clip_batch


class UcfTempTrans(Dataset):
    """UCF101 Dataset"""
    def __init__(self, data_type, opts, split, sp_transform):
        """
        Args:
            opts   : config options
            train : train for training, val for validation, test for testing
            split : 1,2,3
        Returns:
            (tensor(frames), class_id, total_frames): Shape of tensor C x T x H x W
        """
        self.data_type = data_type
        self.opts = opts
        self.sp_transform = sp_transform

        # indexes for training/test set
        if self.data_type == 'train':
            split_name = 'trainlist0{}.txt'.format(split)
        else:
            split_name = 'testlist0{}.txt'.format(split)

        self.data = []  # (filename, id, total_frames)
        with open(os.path.join(self.opts.annotation_path, split_name), 'r') as f:
            for line in f:
                video_path = os.path.join(self.opts.frame_dir, line.split('.')[0])
                if os.path.exists(video_path):
                    total_frames = len(glob.glob(video_path + '/0*.jpg'))
                    self.data.append((video_path, None, total_frames))
                else:
                    print('{} does not exist'.format(video_path))

    def __len__(self):
        '''
        returns number of test set
        ''' 
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        video_path = video[0]
        total_frames = video[2]
        assert self.opts.task in ['r_cls']
        if self.opts.task == 'r_cls':
            clip_temptrans, label = self.temp_transform_clip(self.opts, video_path, total_frames, self.sp_transform)
            return clip_temptrans, label

    def clip_process(self, clip, sample_size, sp_transform):
        clip_tensor = torch.Tensor(3, len(clip), sample_size, sample_size)
        sp_transform.randomize_parameters()
        for i, img in enumerate(clip):
            clip_tensor[:, i, :, :] = sp_transform(img)

        return clip_tensor

    def temp_transform_clip(self, opts, frame_path, total_frames, sp_transform):
        """
            Chooses a random clip from a video for training/ validation
            Args:
                opts         : config options
                frame_path  : frames of video frames
                total_frames: Number of frames in the video
            Returns:
                list(frames) : random clip (list of frames of length sample_duration)
                from a video for training/ validation
            """
        clip_temp_trans = []
        labels = []
        # choosing a speed, max_speed 2**3 = 8
        max_speed = min(int(np.log2(total_frames/opts.sample_duration)), 3)
        if max_speed > 0:
            speed_label = random.randint(0, max_speed)
        else:
            speed_label = 0
        sample_rate = PACE[speed_label]

        # choosing a random frame
        if 'speed' in opts.temp_transform:
            clip = []
            start_frame = np.random.randint(1, total_frames + 2 - opts.sample_duration * sample_rate)
            index_clip = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            for i in index_clip:
                try:
                    img = Image.open(os.path.join(frame_path, '%05d.jpg' % (start_frame + i)))
                except ValueError:
                    print("Reading img error of {}".format(
                                    os.path.join(frame_path, '%05d.jpg' % (start_frame + i))))
                clip.append(img.copy())
                img.close()
            clip = self.clip_process(clip, opts.sample_size, sp_transform)
            clip_temp_trans.append(clip)
            labels.append(speed_label)

        elif 'random' in opts.temp_transform:
            clip = []
            # sample rate in random transform is 1
            start_frame = np.random.randint(1, total_frames - opts.sample_duration)
            for i in range(0, opts.sample_duration):
                try:
                    img = Image.open(os.path.join(frame_path, '%05d.jpg' % (start_frame + i)))
                except ValueError:
                    print("Reading img error of {}".format(frame_path))
                clip.append(img.copy())
                img.close()
            clip = self.clip_process(clip, opts.sample_size, sp_transform)
            random.shuffle(clip)
            clip_temp_trans.append(np.array(clip))
            labels.append(4)  # class label for random

        elif 'priodic' in opts.temp_transform:
            clip = []
            if max_speed > 0:
                start_frame = np.random.randint(1, total_frames - opts.sample_duration * sample_rate)
            else:
                start_frame = 0
            forward = np.arange(0, (opts.sample_duration - 2) * sample_rate, sample_rate)
            if sample_rate > 1:
                # backward shift anchor
                jitter = 0
            else:
                jitter = 1
            offsets = np.random.uniform(jitter, sample_rate + 1 - jitter)
            backward = np.flip(forward)
            sequence = np.concatenate(forward, offsets + backward, axis=1)
            import pdb; pdb.set_trace()
            s_index = np.random.randint(0, len(sequence) - opts.sample_duration)
            sequence = sequence[s_index:(s_index + opts.sample_duration)]
            for i in sequence:
                try:
                    img = Image.open(os.path.join(frame_path, '%05d.jpg' % (start_frame + i)))
                except ValueError:
                    print("Reading img error of {}".format(
                                os.path.join(frame_path, '%05d.jpg' % (start_frame + i))))
                clip.append(img.copy())
                img.close()
            clip = self.clip_process(clip, opts.sample_size, sp_transform)
            clip_temp_trans.append(np.array(clip))
            labels.append(5)

        elif 'warp' in opts.temp_transform:
            clip = []
            if max_speed > 0:
                offsets = np.random.uniform(1, 2 ** max_speed + 1, size=opts.sample_duration)
                index_clip = np.cumsum(offsets)
                start_frame = np.random.randint(1, total_frames - np.max(index_clip))
            else:
                index_clip = np.arange(0, total_frames)
                np.random.shuffle(index_clip)
                index_clip = index_clip[:opts.sample_duration]
                index_clip = np.sort(index_clip)
            for i in index_clip:
                try:
                    img = Image.open(os.path.join(frame_path, '%05d.jpg' % (start_frame + i)))
                except ValueError:
                    print("Reading img error of {}".format(
                                os.path.join(frame_path, '%05d.jpg' % (start_frame + i))))
                clip.append(img.copy())
                img.close()
            clip = self.clip_process(clip, opts.sample_size, sp_transform)
            clip_temp_trans.append(np.array(clip))
            labels.append(6)
        return clip_temp_trans, labels


class Kin400RepreLMDB(object):
    def __init__(self, data_type, opts, split, sp_transform):
        self.data_type = data_type
        self.opts = opts
        self.sp_transform = sp_transform
        # Loading LMDB
        print('Loading LMDB from %s, split:%s' % (self.opts.lmdb_path, split))
        self.env = lmdb.open(self.opts.lmdb_path, subdir=os.path.isdir(self.opts.lmdb_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            db_order = msgpack.loads(txn.get(b'__order__'))
        
        video_dict = dict(zip([i for i in db_order], ["%09d" % i for i in range(len(db_order))]))
    
        # indexes for training/test set
        if self.data_type == "train":
            split_name = "train_list_label_nframe.txt"
        elif self.data_type == "val":
            split_name = "val_list_label_nframe.txt"
        self.data = []  # (filename, id, total_frames)
        with open(os.path.join(self.opts.annotation_path, split_name), 'r') as f:
            for line in f.readlines():
                split = line.strip().split("  ")
                video_name = split[0].split(".")[0]
                video_key = video_dict[video_name]
                label = split[1]
                total_frames = split[2]
                self.data.append((video_name, video_key.encode("ascii"), int(label), int(total_frames)))
        db_order.clear()
        video_dict.clear()

    def pil_from_raw_rgb(self, raw):
        return Image.open(BytesIO(raw)).convert('RGB')

    def __getitem__(self, idx):
        video_name, video_key, _, total_frames = self.data[idx]

        assert self.opts.task in ['r_byol', "loss_com"]
        clip_1, clip_2, tem_label, pb_label, rotate_label = self.repre_train_clip(video_name, video_key, total_frames)
        clip_cat = clip_1 + clip_2
        clip_1, clip_2, spa_label = self.sp_transform(clip_cat)
        return [torch.stack(clip_1).transpose(0, 1), torch.stack(clip_2).transpose(0, 1)], \
               [spa_label, tem_label, pb_label, rotate_label]

    def repre_train_clip(self, video_name, video_key, total_frames):
        """
            Chooses a random clip from a video for training/ validation
            Args:
                frame_path  : frames of video frames
                total_frames: Number of frames in the video
            Returns:
                list(frames) : random clip
        """
        clip_1 = []
        clip_2 = []
        # label = random.randint(0, 3)  # temporal random
        # sample_rate = PACE[label]
        max_pb = int(np.log2(total_frames / (self.opts.sample_duration - 1)))
        pb_label = random.randint(0, min(3, max_pb))
        sample_rate = PACE[pb_label]
        # tem_interval = 0.2
        clip_range = (self.opts.sample_duration - 1) * sample_rate

        rot_label_1 = random.randint(0, 3)
        rotate_angle_1 = ROTATE[rot_label_1]
        rot_label_2 = random.randint(0, 3)
        rotate_angle_2 = ROTATE[rot_label_2]
        # choosing a random frame
        if total_frames - clip_range <= 0:
            index_clip = []
            idx_frame = 0
            while len(index_clip) < self.opts.sample_duration:
                index_clip.append(idx_frame)
                idx_frame += sample_rate
                if idx_frame >= total_frames:
                    idx_frame = 0
            start_frame = 0

            with self.env.begin(write=False) as txn:
                raw = msgpack.loads(txn.get(video_key))

            for i in index_clip:
                try:
                    img = self.pil_from_raw_rgb(raw[start_frame + i])
                    if rotate_angle_1 != 0:
                        img_1 = img.transpose(rotate_angle_1)
                    else:
                        img_1 = img
                    if rotate_angle_2 != 0:
                        img_2 = img.transpose(rotate_angle_2)
                    else:
                        img_2 = img
                except ValueError:
                    print("Reading img error of {}".format(os.path.join(video_name, "%05d.jpg" % (start_frame + i))))
                clip_1.append(img_1.copy())
                clip_2.append(img_2.copy())
                img.close()
            tem_label = 0
            return clip_1, clip_2, tem_label, pb_label, [rot_label_1, rot_label_2]
        else:
            start_frame = random.randint(0, total_frames - clip_range - 1)
            while True:
                tem_label = random.randint(0, 4)
                tem_rate = OVERLAP_TEM_RATE[tem_label]
                front_behind = random.randint(0, 1)  # front of behind of second clip, 0 for front, 1 for behind
                if front_behind == 0:
                    start_frame_2 = start_frame - int((1 - tem_rate) * clip_range)
                    if start_frame_2 < 1:
                        continue
                else:
                    start_frame_2 = start_frame + int((1 - tem_rate) * clip_range)
                    if start_frame_2 > total_frames - clip_range:
                        continue

                index_clip = np.arange(0, clip_range + 1, sample_rate)

                with self.env.begin(write=False) as txn:
                    raw = msgpack.loads(txn.get(video_key))
                for i in index_clip:
                    try:
                        img = self.pil_from_raw_rgb(raw[start_frame + i])
                        if rotate_angle_1 != 0:
                            img = img.transpose(rotate_angle_1)
                    except ValueError:
                        print("Reading img error of {}".format(os.path.join(video_name, "%05d.jpg" % (start_frame + i))))
                    clip_1.append(img.copy())
                    img.close()
                
                with self.env.begin(write=False) as txn:
                    raw = msgpack.loads(txn.get(video_key))

                for i in index_clip:
                    try:
                        img = self.pil_from_raw_rgb(raw[start_frame + i])
                        if rotate_angle_2 != 0:
                            img = img.transpose(rotate_angle_2)
                    except ValueError:
                        print("Reading img error of {}".format(os.path.join(video_name, "%05d.jpg" % (start_frame_2 + i))))
                    clip_2.append(img.copy())
                    img.close()

                return clip_1, clip_2, tem_label, pb_label, [rot_label_1, rot_label_2]

    def __len__(self):
        return len(self.data)


class Kin400FTOfflineLMDB(Dataset):
    def __init__(self, data_type, opts, split, sp_transform):
        """
        Args:
            opts   : config options
            train : train for training, val for validation, test for testing
            split : 1,2,3
        Returns:
            (tensor(frames), class_id, total_frames): Shape of tensor C x T x H x W
        """
        self.data_type = data_type
        self.opts = opts
        self.sp_transform = sp_transform
        # Loading LMDB
        print('Loading LMDB from %s, split:%s' % (self.opts.lmdb_path, split))
        self.env = lmdb.open(self.opts.lmdb_path, subdir=os.path.isdir(self.opts.lmdb_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            db_order = msgpack.loads(txn.get(b'__order__'))
        
        video_dict = dict(zip([i for i in db_order], ["%09d" % i for i in range(len(db_order))]))
    
        # indexes for training/test set
        if self.data_type == "train":
            split_name = "train_list_label_nframe.txt"
        elif self.data_type == "val":
            split_name = "val_list_label_nframe.txt"
        self.data = []  # (filename, id, total_frames)
        with open(os.path.join(self.opts.annotation_path, split_name), 'r') as f:
            for line in f.readlines():
                split = line.strip().split("  ")
                video_name = split[0].split(".")[0]
                video_key = video_dict[video_name]
                label = split[1]
                total_frames = split[2]
                self.data.append((video_name, video_key.encode("ascii"), int(label), int(total_frames)))
        db_order.clear()
        video_dict.clear()

    def __len__(self):
        '''
        returns number of test set
        '''
        return len(self.data)

    def __getitem__(self, idx):
        video_name, video_key, label, total_frames = self.data[idx]

        assert self.opts.task in ["scratch", "ft_all", "ft_fc"]
        if self.data_type == "train":
            clip = self._get_train_clip(video_name, video_key, total_frames)
            return torch.stack(clip).transpose(0, 1), label
        elif self.data_type == "val":
            clip = self._get_val_clip(video_name, video_key, total_frames)
            return torch.stack(clip).transpose(0, 1), label
        elif self.data_type == "test":
            clip_batch = self._get_test_clip(video_name, video_key, total_frames)
            return np.stack(clip_batch).transpose(0, 2, 1, 3, 4), label

    def _get_train_clip(self, video_name, video_key, total_frames):
        sample_rate = self.opts.pb_rate
        clip = []
        clip_range = (self.opts.sample_duration - 1) * sample_rate
        if total_frames - clip_range <= 0:
            index_clip = []
            idx_frame = 0
            while len(index_clip) < self.opts.sample_duration:
                index_clip.append(idx_frame)
                idx_frame += sample_rate
                if idx_frame >= total_frames:
                    idx_frame = 0
            start_frame = 0
        else:
            start_frame = random.randint(0, total_frames - clip_range - 1)
            index_clip = np.arange(0, clip_range + 1, sample_rate)
        
        with self.env.begin(write=False) as txn:
            raw = msgpack.loads(txn.get(video_key))

        for i in index_clip:
            try: img = self.pil_from_raw_rgb(raw[start_frame + i])
            except ValueError:
                print("Reading img error of {}".format(os.path.join(video_name, "%05d.jpg" % (start_frame + i))))
            clip.append(img.copy())
            img.close()

        clip = self.sp_transform(clip)

        return clip

    def _get_val_clip(self, video_name, video_key, total_frames):
        sample_rate = self.opts.pb_rate
        clip = []
        clip_range = (self.opts.sample_duration - 1) * sample_rate
        if total_frames - clip_range <= 0:
            index_clip = []
            idx_frame = 0
            while len(index_clip) < self.opts.sample_duration:
                index_clip.append(idx_frame)
                idx_frame += sample_rate
                if idx_frame >= total_frames:
                    idx_frame = 0
            start_frame = 0
        else:
            start_frame = random.randint(0, total_frames - clip_range - 1)
            index_clip = np.arange(0, clip_range + 1, sample_rate)
        with self.env.begin(write=False) as txn:
            raw = msgpack.loads(txn.get(video_key))
        for i in index_clip:
            try: img = self.pil_from_raw_rgb(raw[start_frame + i])
            except ValueError:
                print("Reading img error of {}".format(os.path.join(video_name, "%05d.jpg" % (start_frame + i))))
            clip.append(img.copy())
            img.close()

        clip = self.sp_transform(clip)

        return clip

    def _get_test_clip(self, video_name, video_key, total_frames):
        clip_batch = []
        sample_rate = self.opts.pb_rate  # can be changed
        clip_range = (self.opts.sample_duration - 1) * sample_rate
        if total_frames - clip_range <= 0:  # pad left, only sample once
            seq_idx = []
            idx_frame = 1
            while len(seq_idx) < self.opts.sample_duration:
                seq_idx.append(idx_frame)
                idx_frame += sample_rate
                if idx_frame >= total_frames:
                    idx_frame = 1
            seq_idx = np.expand_dims(seq_idx, 0)
        else:
            # overlap 0.5
            # start = np.expand_dims(np.arange(0, available+1, self.opts.sample_duration * sample_rate // 2 - 1), 1)
            start = np.expand_dims(np.arange(0, total_frames - clip_range, clip_range), 1)
            seq_idx = np.expand_dims(np.arange(self.opts.sample_duration) * sample_rate, 0) + start
            last = np.expand_dims(np.arange(total_frames - clip_range - 1, total_frames, sample_rate), 0)
            seq_idx = np.append(seq_idx, last, 0)
        
        with self.env.begin(write=False) as txn:
            raw = msgpack.loads(txn.get(video_key))

        for idx in seq_idx:
            clip = []
            for i in idx:
                try: img = self.pil_from_raw_rgb(raw[i])
                except ValueError:
                    print("Reading img error of {}".format(os.path.join(video_name, "%05d.jpg" % i)))
                clip.append(img.copy())
                img.close()

            clip = self.sp_transform(clip)
            clip = torch.stack(clip)
            clip_batch.append(clip)

        return clip_batch

    def pil_from_raw_rgb(self, raw):
        return Image.open(BytesIO(raw)).convert('RGB')


class KINFTOffline(Dataset):
    def __init__(self, data_type, opts, split, sp_transform):
        """
        Args:
            opts   : config options
            train : train for training, val for validation, test for testing
            split : 1,2,3
        Returns:
            (tensor(frames), class_id, total_frames): Shape of tensor C x T x H x W
        """
        self.data_type = data_type
        self.opts = opts
        self.sp_transform = sp_transform
        # indexes for training/test set
        if self.data_type == "train":
            split_name = "train_list_label_nframe.txt"
        elif self.data_type == "val":
            split_name = "val_list_label_nframe.txt"
        self.data = []  # (filename, id, total_frames)
        with open(os.path.join(self.opts.annotation_path, split_name), 'r') as f:
            for line in f.readlines():
                line_split = line.strip().split("  ")
                video_path = os.path.join(self.opts.frame_dir, line_split[0].split('.')[0])
                total_frames = line_split[2]
                self.data.append((video_path, int(line_split[1]), int(total_frames)))
                # if os.path.exists(video_path):
                #     total_frames = line_split[2]
                #     self.data.append((video_path, int(line_split[1]), int(total_frames)))
                # else:
                #     print('{} does not exist'.format(video_path))

    def __len__(self):
        '''
        returns number of test set
        '''
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        video_path = video[0]
        label = video[1]
        total_frames = video[2]
        assert self.opts.task in ["scratch", "ft_all", "ft_fc"]
        if self.data_type == "train":
            clip = self._obtain_train_clip(self.opts, video_path, total_frames, self.sp_transform)
        elif self.data_type == "val":
            clip = self._obtain_val_clip(self.opts, video_path, total_frames, self.sp_transform)
        elif self.data_type == "test":
            clip = self._obtain_test_clip(self.opts, video_path, total_frames, self.sp_transform)
        return clip, label

    def _obtain_train_clip(self, opts, frame_path, total_frames, sp_transform):
        # random spatial cropping + temporal jittering
        clip = []
        # label = random.randint(0, 3)  # temporal random
        # sample_rate = PACE[label]
        sample_rate = opts.pb_rate

        # choosing a random frame
        if total_frames - opts.sample_duration * sample_rate < 0:
            sequence = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip = np.zeros_like(sequence)
            sequence = sequence[sequence < total_frames]
            index_clip[-len(sequence):] = sequence
            start_frame = 1
        else:
            start_frame = np.random.randint(1, total_frames + 2 - opts.sample_duration * sample_rate)
            index_clip = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
        for i in index_clip:
            # try:
            #     img = Image.open(os.path.join(frame_path, '%05d.jpg' % (start_frame + i)))
            # except ValueError:
            #     print("Reading img error of {}".format(
            #                     os.path.join(frame_path, '%05d.jpg' % (start_frame + i))))
            # clip.append(img.copy())
            # img.close()
            try:
                img = cv2.imread(os.path.join(frame_path, '%05d.jpg' % (start_frame + i)))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                clip.append(img)
            except ValueError:
                raise "Reading img error of {}".format(os.path.join(frame_path, '%05d.jpg' % (start_frame + i)))

        clip = np.array(clip)
        #  spacial data augment
        clip = sp_transform(clip)
        return clip

    def _obtain_val_clip(self, opts, frame_path, total_frames, sp_transform):
        """
            Chooses a random clip from a video for training/ validation
            Args:
                opts         : config options
                frame_path  : frames of video frames
                total_frames: Number of frames in the video
            Returns:
                list(frames) : random clip
        """
        clip = []
        # pb_label = 0  # temporal random
        # sample_rate = PACE[pb_label]
        sample_rate = opts.pb_rate

        # choosing a random frame
        if total_frames - opts.sample_duration * sample_rate < 0:
            sequence = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip = np.zeros_like(sequence)
            sequence = sequence[sequence < total_frames]
            index_clip[-len(sequence):] = sequence
            start_frame = 1
        else:
            start_frame = np.random.randint(1, total_frames + 2 - opts.sample_duration * sample_rate)
            index_clip = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
        for i in index_clip:
            try:
                img = cv2.imread(os.path.join(frame_path, '%05d.jpg' % (start_frame + i)))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                clip.append(img)
            except ValueError:
                raise "Reading img error of {}".format(os.path.join(frame_path, '%05d.jpg' % (start_frame + i)))

        clip = np.array(clip)
        #  spacial data augment
        clip = sp_transform(clip)
        return clip

    def _obtain_test_clip(self, opts, frame_path, total_frames, sp_transform):
        pb_rate = opts.pb_rate
        vr = decord.VideoReader(frame_path)
        total_frames = len(vr)
        clip_duration = opts.sample_duration * pb_rate

        if total_frames - clip_duration <= 0:
            sequence = np.arange(0, clip_duration, pb_rate)
            clip_index = np.zeros_like(sequence)
            sequence = sequence[sequence < total_frames]
            clip_index[-len(sequence):] = sequence
            clip_batch = [vr.get_batch(clip_index).asnumpy()]
        else:
            num_batch = int(total_frames / clip_duration)
            clip_index_batch = [list(np.arange(i*clip_duration, (i+1)*clip_duration, pb_rate)) for i in range(num_batch)]
            # append the last continue clip
            clip_index_batch.append(list(np.arange(total_frames-1-clip_duration, total_frames-1, pb_rate)))
            clip_batch = [vr.get_batch(clip_index).asnumpy() for clip_index in clip_index_batch]

        clip_batch = [sp_transform(clip) for clip in clip_batch]
        clip_batch_array = np.array(clip_batch)
        return clip_batch_array


class KINFTOnlineDecord(Dataset):
    def __init__(self, data_type, opts, split, sp_transform):
        """
        Args:
            opts   : config options
            train : train for training, val for validation, test for testing
            split : 1,2,3
        Returns:
            (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.data_type = data_type
        self.opts = opts
        self.sp_transform = sp_transform
        self.toPIL = transforms.ToPILImage()  # convert tensor CHW or np HWC -> PIL

        if self.data_type == 'train':
            split_name = 'train_list_label.txt'
        elif self.data_type == "val":
            split_name = 'val_list_label.txt'

        self.data = []  # (filename, id)
        with open(os.path.join(self.opts.annotation_path, split_name), 'r') as f:
            for line in f:
                line_split = line.strip().split("  ")
                video_path = os.path.join(self.opts.frame_dir, line_split[0])
                if os.path.exists(video_path):
                    self.data.append((video_path, int(line_split[1])))
                else:
                    print('{} does not exist'.format(video_path))

    def __len__(self):
        '''
        returns number of test set
        '''
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        video_path = video[0]
        label_id = video[1]

        if self.data_type == 'test':
            clip_batch = get_test_clip_coclr(self.opts, video_path, self.sp_transform)
            return np.stack(clip_batch).transpose(0, 2, 1, 3, 4), label_id
            # clip = get_test_clip_tf(self.opts, video_path, total_frames, self.sp_transform)
        elif self.data_type == 'val':
            clip = self.get_val_clip(self.opts, video_path)
            clip = self.sp_transform(clip)
        elif self.data_type == 'train':
            clip = self.get_train_clip(self.opts, video_path)
            clip = self.sp_transform(clip)

        return torch.stack(clip).transpose(0, 1), label_id

    def get_train_clip(self, opts, frame_path):
        sample_rate = opts.pb_rate
        clip = []
        vr = decord.VideoReader(frame_path)
        total_frames = len(vr)

        if total_frames - opts.sample_duration * sample_rate <= 0:
            sequence = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip = np.zeros_like(sequence)
            sequence = sequence[sequence < total_frames]
            index_clip[-len(sequence):] = sequence
            start_frame = 1
        else:
            start_frame = np.random.randint(0, total_frames - opts.sample_duration * sample_rate)
            index_clip = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip += start_frame

        clip_numpy = vr.get_batch(index_clip).asnumpy()
        for frame in clip_numpy:
            img = self.toPIL(frame)
            clip.append(img)
        return clip

    def get_val_clip(self, opts, frame_path):
        sample_rate = opts.pb_rate
        clip = []
        vr = decord.VideoReader(frame_path)
        total_frames = len(vr)

        if total_frames - opts.sample_duration * sample_rate < 0:
            sequence = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip = np.zeros_like(sequence)
            sequence = sequence[sequence < total_frames]
            index_clip[-len(sequence):] = sequence
            start_frame = 1
        else:
            start_frame = np.random.randint(0, total_frames - opts.sample_duration * sample_rate)
            index_clip = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip += start_frame

        clip_numpy = vr.get_batch(index_clip).asnumpy()
        for frame in clip_numpy:
            img = self.toPIL(frame)
            clip.append(img)
        return clip


class KINFTOnline(Dataset):
    """kinetics Dataset"""
    def __init__(self, data_type, opts, split, sp_transform):
        """
        Args:
            opts   : config options
            train : train for training, val for validation, test for testing
            split : 1,2,3
        Returns:
            (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.data_type = data_type
        self.opts = opts
        self.sp_transform = sp_transform
        self.toPIL = transforms.ToPILImage()  # convert tensor CHW or np HWC -> PIL

        if self.data_type == 'train':
            split_name = 'train_list_label.txt'
        elif self.data_type == "val":
            split_name = 'val_list_label.txt'
        elif self.data_type == "test":
            split_name = 'val_list_label.txt'

        self.data = []  # (filename, id)
        with open(os.path.join(self.opts.annotation_path, split_name), 'r') as f:
            for line in f:
                line_split = line.strip().split("  ")
                video_path = os.path.join(self.opts.frame_dir, line_split[0])
                self.data.append((video_path, int(line_split[1])))
                # if os.path.exists(video_path):
                #     self.data.append((video_path, int(line_split[1])))
                # else:
                #     print('{} does not exist'.format(video_path))

    def __len__(self):
        '''
        returns number of test set
        '''
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        video_path = video[0]
        label_id = video[1]

        if self.data_type == 'test':
            clip = self.get_test(self.opts, video_path, self.sp_transform)
            # clip = get_test_clip_tf(self.opts, video_path, total_frames, self.sp_transform)
        elif self.data_type == 'val':
            clip = self.get_val_clip(self.opts, video_path)
            clip = self.sp_transform(clip)
        elif self.data_type == 'train':
            clip = self.get_train_clip(self.opts, video_path)
            clip = self.sp_transform(clip)

        return clip, label_id

    def get_train_clip(self, opts, video_path):
        sample_rate = opts.pb_rate
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)

        if total_frames - opts.sample_duration * sample_rate <= 0:
            sequence = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip = np.zeros_like(sequence)
            sequence = sequence[sequence < total_frames]
            index_clip[-len(sequence):] = sequence
            start_frame = 1
        else:
            start_frame = np.random.randint(0, total_frames - opts.sample_duration * sample_rate)
            index_clip = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip += start_frame

        clip_numpy = vr.get_batch(index_clip).asnumpy()
        return clip_numpy

    def get_val_clip(self, opts, video_path):
        sample_rate = opts.pb_rate
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)

        if total_frames - opts.sample_duration * sample_rate <= 0:
            sequence = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip = np.zeros_like(sequence)
            sequence = sequence[sequence < total_frames]
            index_clip[-len(sequence):] = sequence
            start_frame = 1
        else:
            start_frame = np.random.randint(0, total_frames - opts.sample_duration * sample_rate)
            index_clip = np.arange(0, opts.sample_duration * sample_rate, sample_rate)
            index_clip += start_frame

        clip_numpy = vr.get_batch(index_clip).asnumpy()
        return clip_numpy

    def get_test(self, opts, video_path, sp_transform):
        pb_rate = opts.pb_rate
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        clip_duration = opts.sample_duration * pb_rate

        if total_frames - clip_duration <= 0:
            sequence = np.arange(0, clip_duration, pb_rate)
            clip_index = np.zeros_like(sequence)
            sequence = sequence[sequence < total_frames]
            clip_index[-len(sequence):] = sequence
            clip_batch = [vr.get_batch(clip_index).asnumpy()]
        else:
            num_batch = int(total_frames / clip_duration)
            clip_index_batch = [list(np.arange(i*clip_duration, (i+1)*clip_duration, pb_rate)) for i in range(num_batch)]
            # append the last continue clip
            clip_index_batch.append(list(np.arange(total_frames-1-clip_duration, total_frames-1, pb_rate)))
            clip_batch = [vr.get_batch(clip_index).asnumpy() for clip_index in clip_index_batch]

        clip_batch = [sp_transform(clip) for clip in clip_batch]
        clip_batch_array = np.array(clip_batch)
        return clip_batch_array


    # def read_video(self, cap, index_clip):
    #     clip = []
    #     for i in range(index_clip[-1] + 1):
    #         (grabbed, frame) = cap.read()
    #         if i in index_clip:
    #             j = 1  # in case video shorter than sample duration
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             clip.append(frame)
    #             index = index_clip.index(i)
    #             while index+j != len(index_clip) and i == index_clip[index+j]:
    #                 clip.append(frame)
    #                 j += 1
    #     cap.release()
    #     return np.array(clip)
