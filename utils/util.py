import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from imgaug import augmenters as iaa
import numpy as np
import cv2
import matplotlib.pyplot as plt

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def zoom_img(img):
    zoom = iaa.Affine(scale=(0.9, 1.3))
    return zoom.augment_image(img)


def trans_img(img):
    trans = iaa.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)})
    return trans.augment_image(img)


def img_random_brightness(img):
    brightness = iaa.Multiply((0.5, 1.2))
    return brightness.augment_image(img)


def img_flip(img, steering_angle):
    img = cv2.flip(img, 1)
    steering_angle = -steering_angle
    return img, steering_angle


def img_motion_blur(img):
    # 高斯模糊的数据增强器
    motion_blur = iaa.MotionBlur()
    return motion_blur.augment_image(img)


# 自定义的算法组合，数据增强器,path  图片的路径,steering_angle 方向盘的角度
def random_augment(image, steering_angle):
    if np.random.rand() < 0.5:
        image = zoom_img(image)
    if np.random.rand() < 0.5:
        image = trans_img(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() < 0.5:
        image = img_motion_blur(image)
    #     if steering_angle < -0.2:
    #         if np.random.rand()<0.6:
    #             image,steering_angle = img_flip(image,steering_angle)
    #     else:
    if np.random.rand() < 0.2:
        image, steering_angle = img_flip(image, steering_angle)

    return image, steering_angle


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
