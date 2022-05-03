from __future__ import division
from PIL import Image, ImageFilter, ImageOps, ImageChops
import numpy as np
#from torchvision.transforms import *
import torch
import random
import numbers
from torchvision import transforms
from PIL import ImageFilter
import math
import torchvision.transforms.functional as F
import torchvision
import collections
import cv2

scale_choice = [1, 1/2**0.25, 1/2**0.5, 1/2**0.75, 0.5]
crop_positions = ['c', 'tl', 'tr', 'bl', 'br']
OVERLAP_SPA_RATE = [1.0, 0.8, 0.6, 0.4, 0.2]

# def get_transforms(opts):
#     val_transform = None
#     # hyperparameters follow imagenet
#     if opts.task in ['r_byol']:
#         train_transform = Compose([
#                     ClipRandomSizedCrop(size=opts.sample_size, sample_duration=opts.sample_duration, bottom_area=0.2),
#                     transforms.RandomApply([
#                         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#                     ], p=0.8),
#                     transforms.RandomGrayscale(p=0.2),
#                     transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
#                     ClipRandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
#                 ])
#     elif opts.task in ['ft_all', 'ft_fc']:
#         train_transform = Compose([
#                     transforms.RandomResizedCrop(opts.sample_size, scale=(0.2, 1.)),
#                     transforms.RandomApply([
#                         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#                     ], p=0.8),
#                     transforms.RandomGrayscale(p=0.2),
#                     transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
#                 ])
#         val_transform = Compose([
#                     transforms.Resize(128),
#                     transforms.CenterCrop(128),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
#                 ])
#     return train_transform, val_transform


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        # if pic.mode == 'RGB':
        #     img = torch.from_numpy(np.array(pic, np.float32).transpose(2, 0, 1))
        # else:
        #     print('Not supported image mode')
        if pic.mode == 'RGB':
            nchannel = 3
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], nchannel)
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
            if isinstance(img, torch.ByteTensor):
                return img.float().div(self.norm_value)
            else:
                return img
        else:
            print('Not supported image mode')

    def randomize_parameters(self):
        pass


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output sizeself.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.crop_position = self. will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size,
                          int) or (isinstance(size, collections.Iterable) and
                                   len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

    def randomize_parameters(self):
        pass


class RandomResize(object):
    def __init__(self, interpolation=Image.BILINEAR):
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        return img.resize(self.size, self.interpolation)

    def randomize_parameters(self):
        ratio = random.random()
        size = int(256 + ratio * (320 - 256))
        self.size = (size, size)


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled. (W x H)
        Returns:
            PIL.Image: Resized image.
        """
        return img.resize(self.size, self.interpolation)

    def randomize_parameters(self):
        pass


class ColorJittering(object):
    def __init__(self, b=0.8, c=0.8, s=0.8, h=0.2, p=0.8):
        c_jitter = transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)
        self.color_jitter = transforms.RandomApply([c_jitter], p=p)

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Resized image.
        """
        return self.color_jitter(img)

    def randomize_parameters(self):
        pass


class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self):
        pass    


class RandomCrop(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        crop_size = self.size

        image_width = img.size[0]
        image_height = img.size[1]

        x1 = self.tl_x * (image_width - crop_size)
        y1 = self.tl_y * (image_height - crop_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        img = img.crop((x1, y1, x2, y2))

        return img

    def randomize_parameters(self):
        self.tl_x = random.random()
        self.tl_y = random.random()


class MultiScaleCornerCrop(object):
    """Crop the given PIL.Image to randomly selected size.
    A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center.
    This crop is finally resized to given size.
    Args:
        scales: cropping scales of the original size
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.pick_scale)

        image_width = img.size[0]
        image_height = img.size[1]

        if self.pick_loc == 'c':
            center_x = image_width // 2
            center_y = image_height // 2
            box_half = crop_size // 2
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
        elif self.pick_loc == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
        elif self.pick_loc == 'tr':
            x1 = image_width - crop_size
            y1 = 0
            x2 = image_width
            y2 = crop_size
        elif self.pick_loc == 'bl':
            x1 = 0
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
        elif self.pick_loc == 'br':
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.pick_loc = crop_positions[random.randint(0, len(crop_positions)-1)]
        self.pick_scale = scale_choice[random.randint(0, len(scale_choice)-1)]


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std, flag="tf"):
        self.mean = mean
        self.std = std
        self.flag = flag

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """

        if self.flag == "tf":
            tensor = tensor * 2.0 - 1.0
            tensor = torch.clamp(tensor, -1.0, 1.0)
            return tensor
        elif self.flag == "imagenet":
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
            return tensor

    def randomize_parameters(self):
        pass


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.flip_prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def randomize_parameters(self):
        self.flip_prob = random.random()


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_mean(dataset='HMDB51'):
    # assert dataset in ['activitynet', 'kinetics']
    if dataset == 'activitynet':
        return [114.7748, 107.7354, 99.4750]
    elif dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        return [110.63666788, 103.16065604,  96.29023126]
    elif dataset == "HMDB51":
        return [0.36410178082273*255, 0.36032826208483*255, 0.31140866484224*255]


def get_std(dataset='HMDB51'):
    # Kinetics (10 videos for each class)
    if dataset == 'kinetics':
        return [38.7568578, 37.88248729, 40.02898126]
    elif dataset == 'HMDB51':
        return [0.20658244577568*255, 0.20174469333003*255, 0.19790770088352*255]


def clip_process(clip, sp_transform, sample_size):
    clip_tensor = torch.Tensor(3, len(clip), sample_size, sample_size)
    sp_transform.randomize_parameters()
    for i, img in enumerate(clip):
        clip_tensor[:, i, :, :] = sp_transform(img)

    return clip_tensor


class ClipToTensor:
    def __call__(self, imgmap):
        totensor = transforms.ToTensor()
        return [totensor(i) for i in imgmap]


class ClipRandomSizedCrop:
    def __init__(self, size, interpolation=Image.BICUBIC, p=1.0, bottom_area=0.2):
        self.size = size
        self.interpolation = interpolation
        self.threshold = p
        self.bottom_area = bottom_area

    def __call__(self, imgmap):
        img1 = imgmap[0]
        if random.random() < self.threshold:  # do RandomSizedCrop
            for attempt in range(10):
                area = img1.size[0] * img1.size[1]
                target_area = random.uniform(self.bottom_area, 1) * area
                aspect_ratio = random.uniform(3. / 4, 4. / 3)

                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))

                if random.random() < 0.5:
                    w, h = h, w
                if w <= img1.size[0] and h <= img1.size[1]:
                    x1 = random.randint(0, img1.size[0] - w)
                    y1 = random.randint(0, img1.size[1] - h)

                    imgmap = [i.crop((x1, y1, x1 + w, y1 + h)) for i in imgmap]
                    for i in imgmap:
                        assert(i.size == (w, h))

                    return [i.resize((self.size, self.size), self.interpolation) for i in imgmap]

            # Fallback
            scale = ClipScale(self.size, interpolation=self.interpolation)
            crop = ClipCenterCrop(self.size)
            return crop(scale(imgmap))
        else:  # don't do RandomSizedCrop, do CenterCrop
            crop = ClipCenterCrop(self.size)
            return crop(imgmap)


class ClipRandomSizedCropOverlap:
    def __init__(self, size, interpolation=Image.BICUBIC, p=1.0, bottom_area=0.2):
        self.size = size
        self.interpolation = interpolation
        self.threshold = p
        self.bottom_area = bottom_area
        self.overlap_spa = 0
        self.pick_size = []
        self.pick_loc = []

    def __call__(self, imgmap, flag):  # flag=0 for first clip, flag=1 for second clip
        # second clip is decided by the first clip
        img_w = imgmap[0].size[0]
        img_h = imgmap[0].size[1]

        if random.random() < self.threshold:  # do RandomSizedCrop
            while True:
                area = img_w * img_h
                target_area = random.uniform(self.bottom_area, 1) * area
                aspect_ratio = random.uniform(3. / 4, 4. / 3)

                if flag == 0:
                    # first crop
                    w = int(round(math.sqrt(target_area * aspect_ratio)))
                    h = int(round(math.sqrt(target_area / aspect_ratio)))

                    if random.random() < 0.5:
                        w, h = h, w

                    if w <= img_w and h <= img_h:
                        x1 = random.randint(0, img_w - w)
                        y1 = random.randint(0, img_h - h)
                        self.pick_size = [w, h]
                        self.pick_loc = [x1, y1]

                        imgmap = [i.crop((x1, y1, x1 + w, y1 + h)) for i in imgmap]
                        return [i.resize((self.size, self.size), self.interpolation) for i in imgmap]
                elif flag == 1:
                    # second crop
                    p_w, p_h = self.pick_size
                    p_x, p_y = self.pick_loc
                    spa_label = random.randint(0, 4)
                    spa_rate = OVERLAP_SPA_RATE[spa_label]

                    # location of overlap region
                    # four corners 0-left top, 1-right top, 2-left bottom, 3-right bottom
                    corner = random.randint(0, 3)
                    # different corner has different equation
                    if corner == 0:
                        # randomly pick overlap location
                        # overlap w, h
                        s_w = random.randint(int(spa_rate * p_w), p_w)  # when s_h = p_h, s_w is lowest, spa_rate
                        s_h = int(spa_rate * p_w * p_h / s_w)  # set by rate and 1st crop area

                        # right bottom corner
                        e_w = p_x + s_w
                        e_h = p_y + s_h
                        if e_w - p_w >= 0 and e_h - p_h >= 0:
                            imgmap = [i.crop((e_w - p_w, e_h - p_h, e_w, e_h)) for i in imgmap]
                            return [i.resize((self.size, self.size), self.interpolation) for i in imgmap], spa_label
                    elif corner == 1:
                        s_w = random.randint(int(spa_rate * p_w), p_w)
                        s_h = int(spa_rate * p_w * p_h / s_w)
                        e_w = p_x + p_w - s_w + p_w
                        e_h = p_y + s_h
                        if e_w <= img_w and e_h - p_h >= 0:
                            imgmap = [i.crop((e_w - p_w, e_h - p_h, e_w, e_h)) for i in imgmap]
                            return [i.resize((self.size, self.size), self.interpolation) for i in imgmap], spa_label
                    elif corner == 2:
                        s_w = random.randint(int(spa_rate * p_w), p_w)
                        s_h = int(spa_rate * p_w * p_h / s_w)
                        e_w = p_x + s_w
                        e_h = p_y + p_h - s_h + p_h
                        if e_w - p_w >= 0 and e_h <= img_h:
                            imgmap = [i.crop((e_w - p_w, e_h - p_h, e_w, e_h)) for i in imgmap]
                            return [i.resize((self.size, self.size), self.interpolation) for i in imgmap], spa_label
                    elif corner == 3:
                        s_w = random.randint(int(spa_rate * p_w), p_w)
                        s_h = int(spa_rate * p_w * p_h / s_w)
                        e_w = p_x + p_w - s_w + p_w
                        e_h = p_y + p_h - s_h + p_h
                        if e_w <= img_w and e_h <= img_h:
                            imgmap = [i.crop((e_w - p_w, e_h - p_h, e_w, e_h)) for i in imgmap]
                            return [i.resize((self.size, self.size), self.interpolation) for i in imgmap], spa_label
        else:  # don't do RandomSizedCrop, do CenterCrop
            crop = ClipCenterCrop(self.size)
            return crop(imgmap)


class ClipRandomHorizontalFlip:
    def __init__(self, command=None):
        if command == 'left':
            self.threshold = 0
        elif command == 'right':
            self.threshold = 1
        else:
            self.threshold = 0.5

    def __call__(self, imgmap):
        if random.random() < self.threshold:
            return [i.transpose(Image.FLIP_LEFT_RIGHT) for i in imgmap]
        else:
            return imgmap


class ClipColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=1.0, sample_duration=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.threshold = p
        self.sample_duration = sample_duration

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = torchvision.transforms.Compose(transforms)

        return transform

    def __call__(self, imgmap):
        if random.random() < self.threshold:  # do ColorJitter
            transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
            return [transform(i) for i in imgmap]
        else:  # don't do ColorJitter, do nothing
            return imgmap

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class ClipGaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.], sample_duration=0):
        self.sigma = sigma
        self.sample_duration = sample_duration

    def __call__(self, imgmap):
        result = []
        for idx, img in enumerate(imgmap):
            if idx % self.sample_duration == 0:
                sigma = random.uniform(self.sigma[0], self.sigma[1])
            result.append(img.filter(ImageFilter.GaussianBlur(radius=sigma)))
        return result


class ClipRandomGray:
    '''Actually it is a channel splitting, not strictly grayscale images'''
    def __init__(self, p=0.5, dynamic=False, sample_duration=0):
        if sample_duration != 0:
            self.consistent = False
        self.p = p  # prob to grayscale
        self.sample_duration = sample_duration

    def __call__(self, imgmap):
        tmp_p = self.p
        if random.random() < tmp_p:
            return [self.grayscale(i) for i in imgmap]
        else:
            return imgmap

    def grayscale(self, img):
        channel = np.random.choice(3)
        np_img = np.array(img)[:, :, channel]
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
        return img


class TwoClipTransform:
    """Take two random transforms on two clips"""
    def __init__(self, random_crop, base_transform, null_transform, sample_duration, p=0.3):
        # p = probability to use base_transform
        self.random_crop = random_crop
        self.base = base_transform
        self.null = null_transform
        self.p = p
        self.sample_duration = sample_duration  # channel to split the tensor into two

    def __call__(self, x):
        # target: list of image
        assert len(x) == 2 * self.sample_duration

        if random.random() < self.p:
            tr1 = self.base
        else:
            tr1 = self.null

        if random.random() < self.p:
            tr2 = self.base
        else:
            tr2 = self.null

        q = self.random_crop(x[0:self.sample_duration], flag=0)
        q = tr1(q)
        k, spa_label = self.random_crop(x[self.sample_duration::], flag=1)
        k = tr2(k)
        return q, k, spa_label


class OneClipTransform:
    """Take two random transforms on one clips"""
    def __init__(self, base_transform, null_transform, sample_duration):
        self.base = base_transform
        self.null = null_transform
        self.sample_duration = sample_duration  # channel to split the tensor into two

    def __call__(self, x):
        # target: list of image
        assert len(x) == 2 * self.sample_duration

        if random.random() < 0.5:
            tr1, tr2 = self.base, self.null
        else:
            tr1, tr2 = self.null, self.base

        # randomly abandon half
        if random.random() < 0.5:
            xx = x[0:self.sample_duration]
        else:
            xx = x[self.sample_duration::]

        q = tr1(xx)
        k = tr2(xx)
        return q, k


class TransformController:
    def __init__(self, transform_list, weights):
        self.transform_list = transform_list
        self.weights = weights
        self.num_transform = len(transform_list)
        assert self.num_transform == len(self.weights)

    def __call__(self, x):
        idx = random.choices(range(self.num_transform), weights=self.weights)[0]
        return self.transform_list[idx](x)

    def __str__(self):
        string = 'TransformController: %s with weights: %s' % (str(self.transform_list), str(self.weights))
        return string


class ClipNormalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, norm_flag):
        if norm_flag == "tf":
            self.norm = Normalize(None, None, "tf")
        elif norm_flag == "imagenet":
            self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """

        return [self.norm(t) for t in tensor]


class ClipCenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = clip[0].size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return [img.crop((x1, y1, x1 + tw, y1 + th)) for img in clip]


class ClipScale:
    def __init__(self, size, interpolation=Image.BICUBIC):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgmap):
        img1 = imgmap[0]
        if isinstance(self.size, int):
            w, h = img1.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return imgmap
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
        else:
            return [i.resize(self.size, self.interpolation) for i in imgmap]


class NumpyClipResize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, video_clip):
        rsz_video_clip = []
        new_h, new_w = self.output_size

        for frame in video_clip:
            rsz_frame = cv2.resize(frame, (new_w, new_h))
            rsz_video_clip.append(rsz_frame)

        return np.array(rsz_video_clip)


class NumpyClipCenterCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, video_clip):

        h, w = video_clip.shape[1:3]
        new_h, new_w = self.output_size

        h_start = int((h - new_h) / 2)
        w_start = int((w - new_w) / 2)

        center_crop_video_clip = video_clip[:, h_start:h_start + new_h, w_start:w_start + new_w, :]

        return center_crop_video_clip


class NumpyClipScale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, video_clip):
        rsz_video_clip = []
        h, w = video_clip.shape[1:3]
        if h > w:
            ratio = h / w
            new_w = self.output_size
            new_h = int(ratio * new_w)
        else:
            ratio = w / h
            new_h = self.output_size
            new_w = int(ratio * new_h)

        for frame in video_clip:
            rsz_frame = cv2.resize(frame, (new_w, new_h))
            rsz_video_clip.append(rsz_frame)

        return np.array(rsz_video_clip)


class NumpyRandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, video_clip):
        h, w = video_clip.shape[1:3]
        new_h, new_w = self.output_size

        h_start = random.randint(0, h - new_h)
        w_start = random.randint(0, w - new_w)

        rnd_crop_video_clip = video_clip[:, h_start:h_start+new_h, w_start:w_start+new_w, :]

        return rnd_crop_video_clip


class NumpyMultiRatioRandomCrop_old(object):
    def __init__(self, output_size, input_size):
        self.output_size = output_size
        self.input_size = input_size

    def __call__(self, video_clip):
        h, w = video_clip.shape[1:3]
        ratio = self.random_loc()
        new_h = int(self.output_size + ratio * (self.input_size - self.output_size))
        new_w = int(self.output_size + ratio * (self.input_size - self.output_size))

        h_start = random.randint(0, h - new_h)
        w_start = random.randint(0, w - new_w)

        rnd_crop_video_clip = video_clip[:, h_start:h_start+new_h, w_start:w_start+new_w, :]

        return rnd_crop_video_clip

    def random_loc(self):
        ratio = random.random()
        return ratio


class NumpyMultiRatioRandomCrop(object):
    def __init__(self, sample_size, input_size):
        self.sample_size = sample_size
        self.input_size = input_size

    def __call__(self, video_clip):
        h, w = video_clip.shape[1:3]
        ratio = self.random_loc()

        new_h = int(self.input_size * ratio)
        new_w = int(self.input_size * ratio)

        h_start = random.randint(0, h - new_h)
        w_start = random.randint(0, w - new_w)

        rnd_crop_video_clip = video_clip[:, h_start:h_start+new_h, w_start:w_start+new_w, :]

        return rnd_crop_video_clip

    def random_loc(self):
        ratio = random.random()
        # (224/320, 224/256)
        # 0.175 (224/256) -> 0.3
        ratio = 0.7 + 0.3 * ratio
        return ratio


class NumpyClipNorm(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, norm_flag):
        self.norm_flag = norm_flag

    def __call__(self, np_tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if self.norm_flag == "tf":
            np_tensor /= 255
            np_tensor = (np_tensor - 0.5) * 2
        elif self.norm_flag == "imagenet":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            np_tensor = np_tensor / 255
            np_tensor = np_tensor.sub_(mean).div_(std)
            # for i in range(3):
            #     np_tensor[i, :, :, :] = (np_tensor[i, :, :, :] - mean[i]) / std[i]
        return np_tensor


class NumpyHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video_clip):
        if random.random() < self.p:
            # Why copy? Should not, need test, axis 2?
            flip_video_clip = np.flip(video_clip, axis=1).copy()
            return flip_video_clip
        return video_clip


class NumpyToTensor(object):
    def __init__(self):
        super(NumpyToTensor, self).__init__()

    def __call__(self, np_tensor):
        # To C x T x H x W
        np_tensor = np.transpose(np_tensor, [3, 0, 1, 2])
        np_tensor = np.float32(np_tensor)  # To float32
        return np_tensor


class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [skimage.transform.rotate(img, angle) for img in clip]
        elif isinstance(clip[0], Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return rotated


# from CoCLR
def get_transforms(mode, opts):
    if mode in ['pre_train']:
        random_crop = ClipRandomSizedCropOverlap(size=opts.sample_size, bottom_area=0.2)
        null_transform = transforms.Compose([
            ClipRandomHorizontalFlip(),
            ClipToTensor(),
            ClipNormalize(norm_flag="tf")
        ])
        # ClipNormalize_imagenet(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        base_transform = transforms.Compose([
            RandomRotation(10),
            transforms.RandomApply([
                ClipColorJitter(0.4, 0.4, 0.4, 0.1, p=1.0, sample_duration=opts.sample_duration)
                ], p=0.8),
            ClipRandomGray(p=0.2, sample_duration=opts.sample_duration),
            transforms.RandomApply([ClipGaussianBlur([.1, 2.], sample_duration=opts.sample_duration)], p=0.5),
            ClipRandomHorizontalFlip(),
            ClipToTensor(),
            ClipNormalize(norm_flag="tf")
        ])

        # oneclip: temporally take one clip, random augment twice
        # twoclip: temporally take two clips, random augment for each
        sp_transform = TransformController([
                        TwoClipTransform(random_crop, base_transform, null_transform,
                                         sample_duration=opts.sample_duration, p=0.3),
                        OneClipTransform(base_transform, null_transform, sample_duration=opts.sample_duration)],
                        weights=[1, 0])  # pay attention to this, just use two clip
    elif mode in ['img']:
        sp_transform = transforms.Compose([
            ClipRandomSizedCrop(size=opts.sample_size, bottom_area=0.2),
            ClipColorJitter(0.4, 0.4, 0.4, 0.1, p=0.3),
            # ClipRandomHorizontalFlip(),
            ClipToTensor(),
            ClipNormalize(norm_flag="tf")
        ])
    elif mode in ["img_val", "img_test"]:
        if opts.sample_size == 112:
            short_size = 128
        elif opts.sample_size == 224:
            short_size = 256
        sp_transform = transforms.Compose([
            ClipScale(short_size),
            ClipCenterCrop(opts.sample_size),
            ClipToTensor(),
            ClipNormalize(norm_flag="tf")
        ])

    elif mode == 'test_color':
        sp_transform = transforms.Compose([
            ClipScale(opts.sample_size),
            ClipCenterCrop(opts.sample_size),
            ClipColorJitter(0.2, 0.2, 0.2, 0.1, p=0.3),
            ClipToTensor(),
            ClipNormalize(norm_flag="tf")
        ])

    elif mode == "numpy":
        sp_transform = transforms.Compose([
            NumpyMultiRatioRandomCrop(opts.sample_size, opts.input_size),
            NumpyClipResize(opts.sample_size),
            NumpyHorizontalFlip(),
            NumpyToTensor(),
            NumpyClipNorm(norm_flag="tf"),
        ])

    elif mode == "numpy_val":
        sp_transform = transforms.Compose([
            NumpyClipScale(opts.sample_size),
            NumpyClipCenterCrop(opts.sample_size),
            NumpyToTensor(),
            NumpyClipNorm(norm_flag="tf"),
        ])

    print('Transform mode {} is {}'.format(mode, sp_transform))
    return sp_transform
