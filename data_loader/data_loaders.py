import os
import cv2
import imageio
from utils import random_augment
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from base import BaseDataLoader
from PIL import Image
from torchvision import transforms, models, datasets
from typing import Tuple


class AutoPilotData(Dataset):
    def __init__(self, data_dir, roi_h):
        self.data_dir = data_dir
        self.images = os.listdir(self.data_dir)
        self.roi_h = roi_h
        self.transforms = transforms.Compose([
            transforms.Resize([66, 200]),
            # Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
            transforms.ToTensor(),  # 将图片转换为Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [0,1] -> [-1,1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.images[index])
        # image = Image.open(image_path)
        image = cv2.imread(image_path)
        label = float(self.images[index].split('_')[1][:-4])
        image, label = self.preprocess(image, label)
        if self.transforms:
            image = self.transforms(image)

        # print("image:{} label:{}".format(image_path, label))
        return image, label

    def preprocess(self, img, label):
        img = img[self.roi_h:, :, :]
        label = (label - 45.) / 90.
        img, label = random_augment(img, label)
        img = cv2.GaussianBlur(img, (3, 3), 1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        im_pil = im_pil.convert("YCbCr")
        return im_pil, label


class DataLoader(BaseDataLoader):
    """
    End to end autopilot 端到端数据集导入
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, roi_h=340,
                 training=True):
        self.data_dir = data_dir
        self.dataset = AutoPilotData(self.data_dir, roi_h)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
