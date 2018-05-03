#!/usr/bin/env python
# -*- coding: gbk -*-
import os
import pydicom
import numpy as np
import SimpleITK as sitk
from PIL import Image
import cv2
import re

from pydicom import DataElement


def get_dsa_data_element(dsa, param_name):
    try:
        element = dsa.data_element(param_name)
        if element is None:
            return ''
        return element.value
    except KeyError as e:
        return ''


class ReformatDsaFile(object):
    def __init__(self, data_path, result_path, patients_info={}):
        self.data_path = data_path
        self.result_path = result_path
        self.patients_info = patients_info
        self.sop_ids = []
        self.folders = []

    def analyze_direction(self, pp_angle):
        return 'L'

    def get_age(self, info):
        try:
            patient_age = int(re.findall(r'\d+', info)[0])
            return patient_age
        except Exception:
            return None

    def split_patient_name(self, patient_name):
        patient_name = patient_name.replace('/', '')
        patient_info = re.findall(r'[MF]\s?\d+Y', patient_name)
        if patient_info is not None and len(patient_info) != 0:
            patient_info = patient_info[0]
            patient_age = self.get_age(patient_info)
            patient_sex = patient_info[0]
            patient_name = patient_name[:patient_name.index(patient_info)-1]
            return patient_name, patient_age, patient_sex
        else:
            return patient_name, '', ''

    def read_file(self, file_path):
        try:
            dsa = pydicom.read_file(file_path, force=True)
            sopUid = dsa.data_element('SOPInstanceUID').value
            if sopUid in self.sop_ids:
                return

            image_type = dsa.ImageType[0]
            if image_type != 'DERIVED':
                return
            patient_id = dsa.data_element('PatientID').value
            patient_name = str(dsa.data_element('PatientName').value)
            manufacturer = dsa.data_element('Manufacturer').value
            patient_sex = get_dsa_data_element(dsa, 'PatientSex')
            patient_age = get_dsa_data_element(dsa, 'PatientAge')
            if patient_sex == '' or patient_age == '':
                patient_name, patient_age, patient_sex = self.split_patient_name(patient_name)
            else:
                patient_age = self.get_age(patient_age)
                patient_name, _, _ = self.split_patient_name(patient_name)
            study_day = dsa.data_element('StudyDate').value
            pp_angle = get_dsa_data_element(dsa, 'PositionerPrimaryAngle')
            prefix = patient_name.strip().replace(' ', '_') + '_' + study_day
            direction = self.analyze_direction(pp_angle)
            store_path = os.path.join(self.result_path, prefix, direction)
            # image_array = dsa.pixel_array

            image = sitk.ReadImage(file_path)
            image_array = sitk.GetArrayFromImage(image)

            if len(image_array.shape) == 2:
                end_trim = 1
            elif len(image_array.shape) == 3:
                end_trim = image_array.shape[0]
            elif len(image_array.shape) == 4:
                end_trim = image_array.shape[0]
            else:
                raise Exception
            new_patient_info = {}
            new_patient_info['Study Date'] = study_day
            new_patient_info['Patient\'s Name'] = patient_name
            new_patient_info['Patient ID'] = patient_id
            new_patient_info['Manufacturer'] = manufacturer
            new_patient_info['Patient\'s Sex'] = patient_sex
            new_patient_info['Patient\'s Age'] = patient_age
            new_patient_info['Positioner Primary Angle'] = pp_angle
            new_patient_info['SOP Instance UID'] = sopUid
            new_patient_info['Start Trim'] = get_dsa_data_element(dsa, 'StartTrim')
            new_patient_info['Stop Trim'] = get_dsa_data_element(dsa, 'StopTrim')
            if new_patient_info['Start Trim'] == '' or new_patient_info['Stop Trim'] == 0:
                new_patient_info['Start Trim'] = 1
                new_patient_info['Stop Trim'] = end_trim
            else:
                new_patient_info['Start Trim'] = int(new_patient_info['Start Trim'])
                new_patient_info['Stop Trim'] = int(new_patient_info['Stop Trim'])
            new_patient_info['Study Time'] = dsa.data_element('StudyTime').value
            new_patient_info['Modality'] = get_dsa_data_element(dsa, 'Modality')
            new_patient_info['Institution Name'] = get_dsa_data_element(dsa, 'InstitutionName')
            new_patient_info['Recommended Display Frame Rate'] = get_dsa_data_element(dsa, 'RecommendedDisplayFrameRate')
            new_patient_info['Patient\'s Birth Date'] = get_dsa_data_element(dsa, 'PatientBirthDate')
            new_patient_info['Study ID'] = get_dsa_data_element(dsa, 'StudyID')

            # counter = 1
            store_dir = os.path.join(store_path, 'all_image_1')
            if not os.path.exists(store_dir):
                os.makedirs(store_dir)
            else:
                dir_num = len(os.listdir(store_path))
                store_dir = os.path.join(store_path, 'all_image_%d' % (dir_num+1))
                os.makedirs(store_dir)
            image_array = image_array.astype(np.float32)

            print(image_array.shape)
            if len(image_array.shape) == 2:
                max_pixel = np.max(image_array)
                img = image_array * 255 / max_pixel
                image = Image.fromarray(img)
                image = image.convert('L')
                image.save(os.path.join(store_dir, '1.jpeg'))
                # cv2.imwrite(os.path.join(store_dir, '1.jpeg'), img)
            elif len(image_array.shape) == 3:
                for i in range(image_array.shape[0]):
                    max_pixel = np.max(image_array[i])
                    img = image_array[i] * 255 / max_pixel
                    image = Image.fromarray(img)
                    image = image.convert('L')
                    image.save(os.path.join(store_dir, '%d.jpeg' % (i + 1)))
                    # cv2.imwrite(os.path.join(store_dir, '%d.jpeg' % (i + 1)), img)
            elif len(image_array.shape) == 4:
                for i in range(image_array.shape[0]):
                    # max_pixel = np.max(image_array[i])
                    img = image_array[i]
                    cv2.imwrite(os.path.join(store_dir, '%d1.jpeg' % (i + 1)), np.ndarray(img[:,:,2],img[:,:,2],img[:,:,0]))
                    tmp = img[:,:,0]
                    img[:, :, 0] = img[:,:,2]
                    img[:, :, 2] = tmp
                    image = Image.fromarray(img, mode='RGB')
                    # image = image.convert('RGB')
                    # img_bak = np.ndarray(image)
                    # image = image.convert('RGB')
                    image.save(os.path.join(store_dir, '%d.jpeg' % (i + 1)))

            new_patient_info['ImageDir'] = store_dir
            if patient_id in self.patients_info:
                self.patients_info[patient_id].append(new_patient_info)
            else:
                self.patients_info[patient_id] = [new_patient_info,]
            self.sop_ids.append(sopUid)
        except KeyboardInterrupt as e:
            if '.jp' not in file_path:
                print(file_path, repr(e))
            # if os.path.exists(store_dir):
            #     os.removedirs(store_dir)
            return None

    def reformat_file(self, root_path):
        for (root, dir, files) in os.walk(root_path):
            if root not in self.folders:
                self.folders.append(root)
            for file in files:
                self.read_file(os.path.join(root, file))


    def save_info(self):
        print(len(self.folders))
        for path in self.folders:
            print(path)
        import json
        with open(os.path.join(self.result_path, 'patients_info.json'), 'wb') as f:
            f.write((json.dumps(self.patients_info).encode("utf-8")))  # 强制以utf-8转一下byte数据再以普通形式写入 。
            f.close()


if __name__ == '__main__':
    import sys
    # root_path, result_path = sys.argv[1:3]
    root_path = 'E://intern//hospital//data//'
    result_path = 'E://intern//hospital//result1//'


    rdf = ReformatDsaFile(root_path, result_path)
    print(rdf.split_patient_name("WANG_XIUQING^F,49Y^^^_20130808"))

    patient_info = {}

    catCopyIm = Image.open(r'E:\intern\hospital\result1\WANG_LING_HUA_20171010\L\all_image_5\1.jpeg')
    img_array = np.array(catCopyIm)
    rdf.read_file('data/0000')

    # dsa = rdf.reformat_file(root_path)
    rdf.save_info()
