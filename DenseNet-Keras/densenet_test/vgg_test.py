# from keras.applications.vgg16 import VGG16
# from keras import models
# from keras import layers
# from keras import optimizers
# from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import PIL
from PIL import Image
import cv2
img = cv2.imread("3.jpeg", cv2.IMREAD_GRAYSCALE)
# arr = img_to_array(img)
img_shape = img.shape
imgs = np.zeros(shape=(img_shape[0], img_shape[1],3), dtype=np.float32)
imgs[:,:,0]=img[:,:]
imgs[:,:,1]=img[:,:]
imgs[:,:,2]=img[:,:]
# img = array_to_img(imgs)
# cv2.imshow("result", img)
# plt.imshow(dist, cmap='gray')
# plt.show()
# cv2.waitKey(0)
cv2.imwrite('s5.jpg', imgs)
print(imgs)