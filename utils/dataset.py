#!/usr/bin/env python
# encoding: utf-8
"""
@version: JetBrains PyCharm 2017.3.2 x64
@author: baobeila
@contact: baibei891@gmail.com
@software: PyCharm
@file: dataset.py
@time: 2019/12/30 15:28
"""
"""没有使用数据增强"""
import os
from keras.preprocessing import image
import numpy as np
from keras.utils import np_utils
def label_smoothing(input_s, epsilon=0.1):
    a = input_s.shape
    m1 = a[1]
    return ((1-epsilon) * input_s) + (epsilon / m1)


def load_data(path,image_size=(64, 64)):
    # 从该路径下划分训练集与验证集
    files = os.listdir(path)
    images = []
    labels = []

    for f in files:
        img_path = path + f
        for subimg_path in os.listdir(img_path):
            file_path = os.path.join(img_path,subimg_path)
            img = image.load_img(file_path, target_size=image_size)
            img_array = image.img_to_array(img)
            images.append(img_array)

            if 'sdi_pre' in f:
                labels.append(0)
            elif 'spdi_pre' in f:
                labels.append(1)
            else:
                labels.append(2)


    data = np.array(images)
    labels = np.array(labels)

    labels = np_utils.to_categorical(labels, 3)
    return data, label_smoothing(labels)


if __name__ == '__main__':
    path =  './data/train/'
    data,label = load_data(path)
    print(data.shape)#(940, 64, 64, 3)
    print(label.shape)#(940, 3)