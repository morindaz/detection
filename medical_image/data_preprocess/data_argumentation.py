#/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def min_max_scaling(img):
    min_pixel = np.min(img)
    max_pixel = np.max(img)
    return (img - min_pixel) / (max_pixel - min_pixel)

class DataArgumentation(object):
    def __init__(self):
        self.rotation_data_generator = ImageDataGenerator(
            rotation_range=6,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.05,
            zoom_range=0.01,
            fill_mode='nearest'
        )

    def do_argumentation(self, image, label, img_num):
        counter = 0
        images = []
        labels = []
        for batch in self.rotation_data_generator.flow(image):
            img = min_max_scaling(batch[0, :, :, 0])
            counter += 1
            images.append(img)
            labels.append(label)
            if counter >= img_num:
                break
        return images, labels


if __name__ == '__main__':
    my_data_argu = DataArgumentation()
    import cv2
    image = cv2.imread('1.jpeg', cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    img = image.reshape((1,) + image.shape + (1,))
    images, labels = my_data_argu.do_argumentation(img, 1, 3)
    print(len(images))
