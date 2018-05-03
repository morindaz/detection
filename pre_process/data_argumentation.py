#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import cv2
import os
import glob
def min_max_scaling(img):
    min_pixel = np.min(img)
    max_pixel = np.max(img)
    return (img - min_pixel) / (max_pixel - min_pixel)

class DataArgumentation(object):
    def __init__(self):
        pass

    def image_data_generator(self):
        rotation_data_generator = ImageDataGenerator(
            rotation_range=8,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        return rotation_data_generator

    def do_argumentation(self, image, label, img_num):
        counter = 0
        images = []
        labels = []
        for batch in self.rotation_data_generator.flow(image):
            if counter >= img_num:
                break
            img = min_max_scaling(batch[0, :, :, 0])
            counter += 1
            images.append(img)
            labels.append(label)
        return images, labels

    def gamma_convert(self, image, gamma):
        img = min_max_scaling(image)
        img = np.power(img, gamma)
        img = min_max_scaling(img)
        return img

    def flip(self, image, code):
        img = cv2.flip(image, code)
        return img

    def process_image(self, img, input_path, output_path, action_name):
        image = cv2.imread(img, 0)
        # flip_image = self.flip(image, 1)
        gamma_image = self.gamma_convert(image,0.5)
        # cv2.imshow('horizontal flip', flip_image)
        # cv2.waitKey(0)
        _,img_name = os.path.split(img)
        img_name_cut =img_name.split(".")
        img_name_changed = img_name_cut[0]+'_'+action_name+'.jpg'
        cv2.imwrite(output_path+"//"+img_name_changed, gamma_image)


if __name__ == '__main__':
    input_dir = "./label/img_origin"
    output_dir = "./label/img_generator"
    origin_img = glob.glob(input_dir+"/*")

    my_data_argu = DataArgumentation()
    data_gen = my_data_argu.image_data_generator()

    image = "./label/img_origin/1.jpg"
    img = load_img(image)
    x = img_to_array(img)
    x = x.reshape((1,)+x.shape)
    i = 0
    for batch in data_gen.flow(x,batch_size=1,save_to_dir=output_dir,save_prefix="gen",save_format="jpg"):
        i +=1
        if i>20:
            break


    # datagen = my_data_argu.image_data_generator()
    # gen_data = datagen.flow_from_directory(input_dir,batch_size=1,shuffle=False,
    #                                        save_to_dir=output_dir,save_prefix='gen',target_size=(750,750))

    # for img in origin_img:
    #     my_data_argu.process_image(img,input_dir,output_dir,"generator")








    # lut = np.zeros(256, dtype=image.dtype)  # 创建空的查找表
    #
    # hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    # cdf = hist.cumsum()  # 计算累积直方图
    # cdf_m = np.ma.masked_equal(cdf, 0)  # 除去直方图中的0值
    # cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())  # 等同于前面介绍的lut[i] = int(255.0 *p[i])公式
    # cdf = np.ma.filled(cdf_m, 0).astype('uint8')  # 将掩模处理掉的元素补为0
    #
    # # 计算
    # result2 = cdf[image]
    # result = cv2.LUT(image, cdf)
    # # 中值滤波
    # img_medianBlur = cv2.medianBlur(result, 5)
    #
    # cv2.imshow("OpenCVLUT", img_medianBlur)
    # cv2.imshow("NumPyLUT", result2)
    # cv2.waitKey(0)

    # img = cv2.imread("1.jpeg", 0)


    # edges = cv2.Canny(img, 0, 200)
    # cv2.imshow('laplacian',edges)
    # cv2.waitKey(0)

    #
    # img = min_max_scaling(image)
    # cv2.imshow('scale image', img)
    # img = np.power(img, 5)
    # img = min_max_scaling(img)
    # cv2.imshow('gamma image', img)
    # _, img = cv2.threshold(img, np.mean(img) ,255, cv2.THRESH_BINARY)
    # cv2.imshow('binary image', img)
    # img_medianBlur = cv2.medianBlur(img.astype(np.uint8), 5)
    #
    # cv2.imshow("OpenCVLUT", img_medianBlur)
    # cv2.waitKey(0)
    # cv2.waitKey(0)
    #
    # img = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=0)
    # cv2.imshow('gaussian',img)
    # img = img * 255
    # gray_lap = cv2.Laplacian(img.astype(np.uint8), cv2.CV_16S, ksize=3)
    # dst = cv2.convertScaleAbs(gray_lap)
    # cv2.imshow('laplacian',dst)
    #
    # cv2.waitKey(0)
    #
    # image = cv2.imread('1.jpeg', cv2.IMREAD_GRAYSCALE)
    # # cv2.imshow('image', image)
    # # cv2.waitKey(0)
    # img = image.reshape((1,) + image.shape + (1,))
    # images, labels = my_data_argu.do_argumentation(img, 1, 113)
    # print(len(images))