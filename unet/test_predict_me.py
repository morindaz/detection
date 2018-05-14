# -*- coding:utf-8 -*-
'''
This file is used for loading the existing hdf5 and then save the predicted images
to the file.
保存一份测试图像的二进制npy文件，一份测试图像的标签，防止label与经过图像分割后
的标签无法对上。
使用训练好的hdf5模型进行预测，将图像存储到对应文件夹
'''
from unet.unet import *
# from data_process import *
# from data import *

def save_img():
    print("array to image")
    imgs = np.load('./test_image/imgs_mask_test.npy')
    imgs_index = np.load('./npydata/imgs_test_index.npy')
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = array_to_img(img)
        img.save("./test_image/%s" % (imgs_index[i]))

mydata = dataProcess(512,512)
imgs_test = mydata.load_test_data()
myunet = myUnet()
model = myunet.get_unet()
model.load_weights('./hdf5/unet_dsa_900.hdf5')
imgs_mask_test = model.predict(imgs_test, batch_size=1,verbose=1)
np.save('./test_image/imgs_mask_test.npy', imgs_mask_test)
save_img()