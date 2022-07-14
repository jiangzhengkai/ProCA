import random
import numpy as np
import numbers
import collections
from PIL import Image

import torchvision
from torchvision.transforms import functional as F
import cv2
from collections.abc import Sequence
import torch

np.random.seed(0)



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, image, label):
        return F.to_tensor(image), F.to_tensor(label).squeeze()


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, label):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, label


class Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size, resize_label=True):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.resize_label = resize_label

    def __call__(self, image, label):
        image = F.resize(image, self.size, Image.BICUBIC)
        if self.resize_label:
            if isinstance(label, np.ndarray):
                # assert the shape of label is in the order of (h, w, c)
                label = cv2.resize(label, (self.size[1], self.size[0]), cv2.INTER_LINEAR)
            else:
                label = F.resize(label, self.size, Image.NEAREST)
        return image, label


class RandomScale(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, scale, size=None, resize_label=True):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        self.scale = scale
        self.size = size
        self.resize_label = resize_label

    def __call__(self, image, label):
        w, h = image.size
        if self.size:
            h, w = self.size
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        size = (int(h * temp_scale), int(w * temp_scale))
        image = F.resize(image, size, Image.BICUBIC)
        if self.resize_label:
            if isinstance(label, np.ndarray):
                # assert the shape of label is in the order of (h, w, c)
                label = cv2.resize(label, (self.size[1], self.size[0]), cv2.INTER_LINEAR)
            else:
                label = F.resize(label, size, Image.NEAREST)
        return image, label


class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, label_fill=255, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        if isinstance(size, numbers.Number):
            self.padding = (padding, padding, padding, padding)
        elif isinstance(size, tuple):
            if padding is not None and len(padding) == 2:
                self.padding = (padding[0], padding[1], padding[0], padding[1])
            else:
                self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.label_fill = label_fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image, label):
        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            if isinstance(label, np.ndarray):
                label = np.pad(label, ((self.padding[1], self.padding[3]), (self.padding[0], self.padding[2]), (0, 0)),
                               mode='constant')
            else:
                label = F.pad(label, self.padding, self.label_fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            if isinstance(label, np.ndarray):
                label = np.pad(label, ((0, 0), (self.size[1] - image.size[0], self.size[1] - image.size[0]), (0, 0)),
                               mode='constant')
            else:
                label = F.pad(label, (self.size[1] - label.size[0], 0), self.label_fill, self.padding_mode)

        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            if isinstance(label, np.ndarray):
                label = np.pad(label, ((self.size[0] - image.size[1], self.size[0] - image.size[1]), (0, 0), (0, 0)),
                               mode='constant')
            else:
                label = F.pad(label, (0, self.size[0] - label.size[1]), self.label_fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)
        image = F.crop(image, i, j, h, w)
        if isinstance(label, np.ndarray):
            # assert the shape of label is in the order of (h, w, c)
            label = label[i:i + h, j:j + w, :]
        else:
            label = F.crop(label, i, j, h, w)
        return image, label

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = F.hflip(image)
            if label is not None:
                if isinstance(label, np.ndarray):
                    # assert the shape of label is in the order of (h, w, c)
                    label = label[:, ::-1, :]
                else:
                    label = F.hflip(label)
        return image, label


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue, )

    def __call__(self, image, label):
        image = self.color_jitter(image)
        return image, label

class RandomApply(object):
    def __init__(self, transforms, p=0.5):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms
        self.p = p

    def __call__(self, image, label):
        if self.p < random.random():
            return image, label
        for t in self.transforms:
            image, label = t(image, label)
        return image, label

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, image, label):
        num_output_channels = 1 if image.mode == 'L' else 3
        if random.random() < self.p:
            return F.to_grayscale(image, num_output_channels=num_output_channels), label
        return image, label

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size



class GaussianBlur(object):
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
        for ks in range(len(self.kernel_size)):
            if self.kernel_size[ks] <= 0:
                raise ValueError("Kernel size value should be an positive number.")
            elif self.kernel_size[ks] % 2 == 0:
                self.kernel_size[ks] += 1

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("If sigma is a single value, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0. < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise ValueError("sigma should be a single number of a list/tuple with length 2.")

        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min, sigma_max):
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def __call__(self, image, label):
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        # TODO need a higher version of torchvision
        return F.gaussian_blur(image, self.kernel_size, [sigma, sigma]), label

    def __repr__(self):
        s = '(kernel_size={}, '.format(self.kernel_size)
        s += 'sigma={})'.format(self.sigma)
        return self.__class__.__name__ + s


class GaussianBlur_simclr(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, image, label):
        image = np.array(image)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), sigma)

        return image, label