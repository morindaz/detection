from unet import *
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