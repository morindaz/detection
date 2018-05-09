#!/usr/bin/env python
# -*- coding: utf-8 -*-
import SimpleITK as sitk


def save_dsa_to_jpg(dsa_path):
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dsa_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        image_array = sitk.GetArrayFromImage(image)  # z, y, x
        origin = image.GetOrigin()  # x, y, z
        spacing = image.GetSpacing()
        # image = sitk.ReadImage(dsa_path)
        # image_array = sitk.GetArrayFromImage(image)
        # for i in range(image_array.shape[0]):
        #     img = Image.fromarray(image_array[i])
        #     img = img.convert('L')
        #     img.save(str(i)+'.jpeg')

        # while True:
        #     store_dir = os.path.join(store_path, 'all_image_%d' % counter)
        #     if not os.path.exists(store_dir):
        #         os.makedirs(store_dir)
        #         break
        #     else:
        #         counter += 1
    except Exception as e:
        print(repr(e))
        return None
        # print(image_array.shape)
    # image_array = sitk.GetArrayFromImage(image)
    # ds = sitk.ReadImage(dsa_path)
    # img_array = sitk.GetArrayFromImage(ds)
    # frame_num, width, height = img_array.shape
    # for i in range(frame_num):
    #     img = img_array[frame_num]
    #     print('i')

import numpy as np
a = np.uint16([118, 255])
print(a * 255)