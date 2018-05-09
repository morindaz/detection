# coding=utf-8
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from matplotlib import pyplot as plt
import numpy as np
import cv2
img = cv2.imread("22.jpeg", 0)
arr = img_to_array(img)
print(arr)
