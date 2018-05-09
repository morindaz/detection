# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import os
import itertools
import numpy as np
# from compiler.ast import flatten
#method 1
x= np.array([[1,2,4],[1],[2],[4],[5]])
out = list(itertools.chain.from_iterable(x))
print(out)

#method 2
final = []
for i in x:
    final.extend(i)
print(final)
#method 3
from compiler.ast import flatten
flatten(a)
# img = cv2.imread(".//mask_gen0_0_840.jpg",0)
# img2 = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
# print(img.shape)
# # change = array_to_img(img)
# cv2.imshow("result", img)
# # plt.imshow(dist, cmap='gray')
# # plt.show()
# cv2.waitKey(0)
# cv2.imshow("result", img2)
# print(img2.shape)
# # plt.imshow(dist, cmap='gray')
# # plt.show()
# cv2.waitKey(0)