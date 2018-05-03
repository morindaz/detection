#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

import cv2
import numpy as np
import keras.backend as K
from keras.utils import np_utils
import os
import pandas as pd

artery_label_dict = {
    'AP_LICA': 0,
    'AP_RICA': 1,
    'AP_LVA': 2,
    'AP_RVA': 3,
    'AP_LECA': 4,
    'AP_RECA': 5,
    'AP_LSUBA': 6,
    'AP_RSUBA': 7,
    'AP_LCCA': 8,
    'AP_RCCA': 9,
    'AP_ARCH': 10,
    'AP_LICA_Z': 11,
    'AP_RICA_Z': 12,
    'LAT_ICA': 0,
    'LAT_VA': 1,
    'LAT_ECA': 2,
    'LAT_CCAp': 3,
    'LAT_CCAd': 4,
    'LAT_ICA_Z': 5
}
image_rows = 224
image_cols = 224


def min_max_scaling(img):
    min_pixel = np.min(img)
    max_pixel = np.max(img)
    return (img - min_pixel) * 255/ (max_pixel - min_pixel)


def convert_artery_label(artery_type):
    if artery_type not in artery_label_dict:
        return None
    return artery_label_dict[artery_type]


def load_image_from_folder(dir, img_type='.jpeg'):
    # dir = os.path.join(dir, 'all_image_1')
    if not os.path.exists(dir):
        print('path: {0} not exist'.format(dir))
        return None
    img_files = os.listdir(dir)
    img_sample = cv2.imread(os.path.join(dir, img_files[0]), cv2.IMREAD_GRAYSCALE)
    img_shape = img_sample.shape
    imgs = np.zeros(shape=(len(img_files), img_shape[0], img_shape[1]), dtype=np.float32)
    for i in range(len(img_files)):
        file_path = os.path.join(dir, str(i + 1) + img_type)
        if os.path.exists(file_path):
            img_data = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            imgs[i] = img_data
        else:
            print('path: {0} not exist'.format(file_path))
            return None
    return imgs


def generate_train_data(image_folder, device_type, resize=None):
    images = load_image_from_folder(image_folder)
    if images is None:
        return
    [img_num, img_w, img_h] = images.shape
    if device_type == 'GE MEDICAL SYSTEMS':
        start_point = int(np.ceil(img_num * 0.2))
        end_point = int(np.ceil(img_num * 0.7))
    else:
        start_point = int(np.ceil(img_num * 0.25))
        end_point = int(np.ceil(img_num * 0.6))

    sp1 = np.zeros((img_w, img_h), dtype=np.float32)
    sp2 = np.zeros((img_w, img_h), dtype=np.float32)
    for m in range(start_point, end_point):
        sp1 = cv2.add(sp1, images[m])
    for n in range(img_num):
        super2 = cv2.add(sp2, images[n])

    if resize is None:
        sp1 = min_max_scaling(sp1 / (end_point - start_point))
        sp2 = min_max_scaling(super2 / (img_num))
        sp3 = min_max_scaling(super2 - (images[0] + images[-1]) / 2)
    else:
        sp1 = min_max_scaling(cv2.resize(sp1 / (end_point - start_point), resize))
        sp2 = min_max_scaling(cv2.resize(super2 / (img_num), resize))
        sp3 = min_max_scaling(cv2.resize(super2 - (images[0] + images[-1]) / 2, resize))
    return sp1, sp2, sp3


def unlabeled_dsa_filter(data):
    data = data[data[1].notna()]
    unlabeled_patient = set(data[data[1] == 'LAT_DSA'][0])
    flag = data.apply(lambda x: x[0] not in unlabeled_patient, axis=1)
    data = data[data.apply(lambda x: x[0] not in unlabeled_patient, axis=1)]
    return data


def read_data(data_file):
    ap_imgs = []
    lat_imgs = []
    ap_img_type = []
    lat_img_type = []
    data = pd.read_excel(data_file, header=None)
    data = unlabeled_dsa_filter(data)
    ap_image_index = 1
    lat_image_index = 1
    for index in data.index:
        artery_type = data.loc[index].values[1]
        artery_file = data.loc[index].values[2]
        device_type = data.loc[index].values[3]
        if 'UnKnown' not in artery_type:
            if not artery_type.startswith('AP_') and not artery_type.startswith('LAT_'):
                artery_type = 'AP_' + artery_type
            # artery_file, artery_type = line.split('|')
            artery_label = convert_artery_label(artery_type)
            if artery_label is None:
                continue
            train1, train2, train3 = generate_train_data(artery_file, device_type)
            if train1 is None:
                continue
            if artery_type.startswith('LAT_'):
                filename = '%d.jpeg' % lat_image_index
                cv2.imwrite(os.path.join('./train_img/lat_train_1', filename), train1)
                cv2.imwrite(os.path.join('./train_img/lat_train_2', filename), train2)
                cv2.imwrite(os.path.join('./train_img/lat_train_3', filename), train3)
                lat_img_type.append([filename, artery_label, artery_file])
                lat_image_index += 1
            else:
                filename = '%d.jpeg' % ap_image_index
                cv2.imwrite(os.path.join('./train_img/ap_train_1', filename), train1)
                cv2.imwrite(os.path.join('./train_img/ap_train_2', filename), train2)
                cv2.imwrite(os.path.join('./train_img/ap_train_3', filename), train3)
                ap_img_type.append([filename, artery_label, artery_file])
                ap_image_index += 1
    ap_img_label = pd.DataFrame(ap_img_type, columns=['filename', 'label', 'dsa_path'])
    ap_img_label.to_excel('./train_img/ap_label.xlsx', index=False)
    lat_img_label = pd.DataFrame(lat_img_type, columns=['filename', 'label', 'dsa_path'])
    lat_img_label.to_excel('./train_img/lat_label.xlsx', index=False)


def create_dir():
    dirs = ['./train_img/ap_train_1', './train_img/ap_train_2', './train_img/ap_train_3',
            './train_img/lat_train_1', './train_img/lat_train_2', './train_img/lat_train_3']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def load_data(data_dir, label_file, resize, num_classes, train_ratio):
    label_data = pd.read_excel(label_file).sample(frac=1)

    img_len = len(label_data)
    imgs = np.zeros(shape=(img_len, resize[0], resize[1], 1), dtype=np.float32)
    img_label = []
    counter = 0
    for index in label_data.index:
        file_name = label_data.loc[index].values[0]
        file_label = label_data.loc[index].values[1]
        img_sample = cv2.imread(os.path.join(data_dir, file_name), cv2.IMREAD_GRAYSCALE)
        img_sample = cv2.resize(img_sample, resize)
        img_sample = img_sample[:, :, np.newaxis]
        imgs[counter] = img_sample
        counter += 1
        if label_file.startswith('ap'):
            if file_label > 10:
                file_label = file_label - 11
        else:
            if file_label == 5:
                file_label = 0
            if file_label == 4:
                file_label = 3
        img_label.append(file_label)

    if K.image_dim_ordering() == 'th':
        imgs = np.array([img.transpose(2, 0, 1) for img in imgs])

    img_labels = np_utils.to_categorical(img_label, num_classes)
    train_len = int(train_ratio * img_len)
    return imgs[:train_len, :, :, :], img_labels[:train_len, :],\
           imgs[train_len:, :, :, :], img_labels[train_len:, :]


# create_dir()
# read_data('E:\intern\hospital\data\label_data\label_file.xlsx')
# x, y, tx, ty = load_data('./train_img/ap_train_1', './train_img/ap_label.xlsx', (224, 224), 13, 0.7)
# print()

def data_argument(img, methods=['flip']):
    arg_data = {}
    for method in methods:
        if method == 'flip':
            arg_data[method] = cv2.flip(img, 1)
        elif method == 'random rotate':
            angle = round(random.random()*3 + 1, 3)
            arg_data[method] = rotate_bound(img, angle)
        elif method == 'translation':
            pass
    return arg_data


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

# image = cv2.imread('E:\\intern\\hospital\\data\\train_img_1\\ap_train_1\\2114.jpeg', cv2.IMREAD_GRAYSCALE)
# # cv2.imshow('image', image)
# # cv2.waitKey(0)
# image_rotate = cv2.resize(rotate_bound(image, 3), (224, 224))
# cv2.imshow('image', image_rotate)
# cv2.waitKey(0)

# def find_bg(image_folder, device_type = 'GE MEDICAL SYSTEMS'):
#     images = load_image_from_folder(image_folder)
#     if images is None:
#         return
#     [img_num, img_w, img_h] = images.shape
#     if device_type == 'GE MEDICAL SYSTEMS':
#         start_point = int(np.ceil(img_num * 0.2))
#         end_point = int(np.ceil(img_num * 0.7))
#     else:
#         start_point = int(np.ceil(img_num * 0.25))
#         end_point = int(np.ceil(img_num * 0.6))
#
#     sp1 = np.zeros((img_w, img_h), dtype=np.float32)
#     for m in range(start_point, end_point):
#         sp1 = cv2.add(sp1, images[m])
#     sp1 = sp1 / (end_point - start_point)
#     diffs = {}
#     for i in range(len(images)):
#         diffs[i] = np.sum(np.abs(cv2.subtract(sp1, images[i])))
#     result = sorted(diffs.items(), lambda x, y: True if x[1] <= y[1] else False)
#     result


def image_binarization(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    ret, image = cv2.threshold(image, np.mean(image), 255, cv2.THRESH_BINARY)
    cv2.imshow('image', image)
    cv2.waitKey(0)

image = cv2.imread('.\\train_img\\ap_train_1\\2114.jpeg', cv2.IMREAD_GRAYSCALE)
image_binarization(image)

