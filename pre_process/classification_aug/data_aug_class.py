'''
For adding the classification training set
'''
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import pandas as pd
import os
import cv2
import numpy as np
import glob
import itertools


class DataAugumentation(object):
    def __init__(self):
        pass
    def image_data_generator(self):
        rotation_data_generator = ImageDataGenerator(
            rotation_range=8,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        return rotation_data_generator


def gen_images(data,type_all,aug_amount,img_input_dir, img_output):
    image_name = data['filename']
    img_type = data['label']
    img_address = data['dsa_path']
    output_name = "aug_labels_0to3.csv"
    add_names = []
    add_type = []
    add_address = []
    img_origin = glob.glob(img_input_dir + "/*")
    my_data_argu = DataAugumentation()
    image_datagen = my_data_argu.image_data_generator()
    for type in type_all:
        for i in range(len(img_type)):
            img_output_dir = img_output + str(type)
            if not os.path.lexists(img_output_dir):
                os.mkdir(img_output_dir)
            if img_type[i] == type:
                image_x = img_input_dir +image_name[i]
                img_x = load_img(image_x)
                x = img_to_array(img_x)
                x = x.reshape((1,) + x.shape)
                count = 0
                for x_batch in image_datagen.flow(x, batch_size=1, save_to_dir=img_output_dir, save_prefix="img_gen" + str(type),
                                                  save_format="jpg"):
                    # add_names.append("added_{}_{}".format(str(count), str(image_name[i])))
                    add_type.append(type)
                    add_address.append(img_output_dir)
                    count += 1
                    if count > aug_amount:
                        break
        img_generated = glob.glob(img_output_dir + "/*")
        if img_generated != []:
            add_names.append(img_generated)
    add_names = list(itertools.chain.from_iterable(add_names))
    for i in range(len(add_names)):
        _, img_name = os.path.split(add_names[i])
        add_names[i] = img_name

    # add_names = np.
    print(len(add_names))
    print(len(add_type))
    print(len(add_address))
    df = {"filename": add_names, "type": add_type, "address": add_address}
    output = pd.DataFrame(df)
    output.to_csv(output_name)
    print("labels generated to file {} finished".format(output_name))




if __name__ == '__main__':

    data_file = "./train_img//ap_label.xlsx"  # label file
    data = pd.read_excel(data_file, header=0)
    img_input_dir = "./test_image//"
    img_output = "./image_gen//"

    type_all = [0,1,2,3]  # The types need to be augumented
    aug_amount = 8  # add images anount
    gen_images(data,type_all,aug_amount,img_input_dir,img_output)



