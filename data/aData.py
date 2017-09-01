from __future__ import division

import math
import os
import os.path
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
from torchvision.transforms import Scale, CenterCrop

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    for bag in sorted(os.listdir(dir)):
        root = os.path.join(dir, bag)
        for fname in sorted(os.listdir(root)):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder_train(data.Dataset):
    def __init__(self, root, transform=None):  # , option=None):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in folders."))

        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        path = self.imgs[index]
        img = loader(path)

        if random.random() < 0.5:  # rotate
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        satRatio = random.uniform(1, 1.2)  # saturation
        img = ImageEnhance.Color(img).enhance(satRatio)
        return self.transform(img)

    def __len__(self):
        return len(self.imgs)


def CreateDataLoader(opt):
    random.seed(opt.manualSeed)

    Trans = transforms.Compose([
        transforms.Scale(64, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_train = ImageFolder_train(root=os.path.join(opt.dataroot), transform=Trans)

    assert dataset_train

    return data.DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers),
                           drop_last=True)
