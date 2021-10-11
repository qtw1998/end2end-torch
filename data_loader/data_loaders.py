import os

import imageio
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from base import BaseDataLoader
from PIL import Image
from torchvision import transforms, models, datasets
from typing import Tuple


class AutoPilotData(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = os.listdir(self.data_dir)
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
        image = Image.open(image_path)
        label = (float(self.images[index].split('_')[1][:-4]) - 45) / 90.0
        if self.transforms:
            image = self.transforms(image)

        # print("image:{} label:{}".format(image_path, label))
        return image, label


class DataLoader(BaseDataLoader):
    """
    End to end autopilot 端到端数据集导入
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = AutoPilotData(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


