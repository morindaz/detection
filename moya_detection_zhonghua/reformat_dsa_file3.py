#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import dicom
import numpy as np
import SimpleITK as sitk
from PIL import Image
import cv2
import re
from moyamoya1_net import moya_diag
import dsa_tools
import json
import traceback

def get_dsa_data_element(dsa, param_name):
    try:
        element = dsa.data_element(param_name)
        if element is None:
            return ''
        return element.value
    except KeyError as e:
        return ''


class ReformatDsaFile(object):
    def __init__(self, data_path, result_path, result, resize, sop_ids=[], patients_info={}):
        self.data_path = data_path
        self.result_path = result_path
        self.resize = resize
        self.patients_info = patients_info
        self.sop_ids = sop_ids
        self.nn = moya_diag()
        self.result = result

    def analyze_direction(self, pp_angle):
        if 15.0 >= pp_angle >= -15.0:
            return 'AP'
        elif 105.0 >= pp_angle >= 75.0:
            return 'LAT'
        else:
            return 'Unknown'

    def get_age(self, info):
        try:
            patient_age = int(re.findall(r'\d+', info)[0])
            return patient_age
        except Exception:
            return None

    def get_philips_age(self, patient_name):
        patient_age = re.findall('\d+', patient_name)[0]
        return patient_age

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
            dsa = dicom.read_file(file_path, stop_before_pixels=True)
            image_type = dsa.ImageType[0]
            if image_type != 'DERIVED':
                return
            sopUid = dsa.data_element('SOPInstanceUID').value
            patient_id = dsa.data_element('PatientID').value
            study_day = dsa.data_element('StudyDate').value
            uniqueid = patient_id + '_' + study_day + '_' + sopUid
            if uniqueid in self.sop_ids:
                return

            pp_angle = float(dsa.data_element('PositionerPrimaryAngle').value)
            position = self.analyze_direction(pp_angle)

            angle_increment = get_dsa_data_element(dsa, 'PositionerPrimaryAngleIncrement')
            if angle_increment != '' and len(angle_increment) > 50:
                print('####### angle increment ######', file_path)
                return

            patient_id = dsa.data_element('PatientID').value
            patient_name = str(dsa.data_element('PatientName').value).replace('/','')
            manufacturer = dsa.data_element('Manufacturer').value
            patient_sex = get_dsa_data_element(dsa, 'PatientSex')
            patient_age = get_dsa_data_element(dsa, 'PatientAge')
            if manufacturer == 'Philips Medical Systems':
                patient_age = self.get_philips_age(patient_name)
            # if patient_sex == '' or patient_age == '':
            #     patient_name, patient_age, patient_sex = self.split_patient_name(patient_name)
            # else:
            #     patient_age = self.get_age(patient_age)
            #     patient_name, _, _ = self.split_patient_name(patient_name)
            patient_prefix = patient_name.strip().replace(' ', '_') + '_' + study_day

            image = sitk.ReadImage(file_path)
            image_array = sitk.GetArrayFromImage(image)

            image_shape = image_array.shape
            if len(image_shape) == 2:
                end_trim = 1
            elif len(image_shape) == 3:
                end_trim = image_shape[0]
            else:
                raise Exception
            new_patient_info = {}
            new_patient_info['height'] = image_shape[1]
            new_patient_info['width'] = image_shape[2]
            new_patient_info['PositionerPrimaryAngle'] = pp_angle
            new_patient_info['ImageType'] = image_type
            new_patient_info['StudyDate'] = study_day
            new_patient_info['PatientName'] = patient_name
            new_patient_info['PatientID'] = patient_id
            new_patient_info['Manufacturer'] = manufacturer
            new_patient_info['PatientSex'] = patient_sex
            new_patient_info['PatientAge'] = patient_age
            new_patient_info['SOPInstanceUID'] = sopUid
            new_patient_info['Position'] = position
            new_patient_info['ArteryType'] = ''
            new_patient_info['StartTrim'] = get_dsa_data_element(dsa, 'StartTrim')
            new_patient_info['StopTrim'] = get_dsa_data_element(dsa, 'StopTrim')
            new_patient_info['InstitutionName'] = get_dsa_data_element(dsa, 'InstitutionName')
            if new_patient_info['StartTrim'] == '' or new_patient_info['StopTrim'] == 0:
                new_patient_info['StartTrim'] = 1
                new_patient_info['StopTrim'] = end_trim
                new_patient_info['NumberOfFrames'] = end_trim
            else:
                new_patient_info['StartTrim'] = int(new_patient_info['StartTrim'])
                new_patient_info['StopTrim'] = int(new_patient_info['StopTrim'])
                new_patient_info['NumberOfFrames'] = new_patient_info['StopTrim']
            new_patient_info['InstanceNumber'] = get_dsa_data_element(dsa, 'InstanceNumber')
            new_patient_info['StudyTime'] = dsa.data_element('StudyTime').value
            new_patient_info['ContentTime'] = dsa.data_element('ContentTime').value
            new_patient_info['Modality'] = get_dsa_data_element(dsa, 'Modality')
            new_patient_info['RecommendedDisplayFrameRate'] = get_dsa_data_element(dsa, 'RecommendedDisplayFrameRate')
            new_patient_info['PatientBirthDate'] = get_dsa_data_element(dsa, 'PatientBirthDate')
            new_patient_info['StudyID'] = get_dsa_data_element(dsa, 'StudyID')


            if position == 'AP':
                dir_prefix = nn.predDicom(frameArray = image_array, manufacturer = manufacturer)
                new_patient_info['ArteryType'] = dir_prefix
            elif position == 'LAT':
                dir_prefix = position
                new_patient_info['ArteryType'] = 'LAT_DSA'
            else:
                dir_prefix = sopUid

            if position != 'Unknown':
                patient_path = os.path.join(self.result_path, patient_prefix)
                store_path = os.path.join(patient_path, '%s_image_1' % dir_prefix)
                if os.path.exists(store_path):
                    dir_num =0
                    for dir_name in os.listdir(patient_path):
                        if dir_name.startswith(dir_prefix):
                            dir_num += 1
                    store_path = os.path.join(patient_path, '%s_image_%d' % (dir_prefix, dir_num + 1))
                    # os.makedirs(store_path)
                # convert_image = dsa_tools.super_diag2(image, manufacturer, self.resize)
                # cv2.imwrite()
            else:
                patient_path = os.path.join(self.result_path, patient_prefix)
                store_path = os.path.join(patient_path, dir_prefix)
                # if os.path.exists(store_path):
                #     return
            convert_image = dsa_tools.super_diag2(image_array, manufacturer, self.resize)
            os.makedirs(store_path)
            convert_image_path = os.path.join(store_path, 'convert_image.jpeg')
            cv2.imwrite(convert_image_path, convert_image)

            if len(image_array.shape) == 3:
                image_num = image_array.shape[0]
                if image_num < 6 or image_num > 50:
                    return
                store_dir = os.path.join(store_path, 'all_image_1')
                store_tiff_dir = os.path.join(store_path, 'tiff_images_1')
                if not os.path.exists(store_dir):
                    os.makedirs(store_dir)
                    os.makedirs(store_tiff_dir)
                else:
                    dir_num = len(os.listdir(store_path))
                    store_dir = os.path.join(store_path, 'all_image_%d' % (dir_num + 1))
                    store_tiff_dir = os.path.join(store_path, 'tiff_images_%d' % (dir_num + 1))
                    os.makedirs(store_dir)
                    os.makedirs(store_tiff_dir)
                image_array = image_array.astype(np.float32)
                for i in range(image_array.shape[0]):
                    max_pixel = np.max(image_array[i])
                    img = image_array[i] * 255 / max_pixel
                    img_tiff = img.astype(np.uint8)
                    # image = Image.fromarray(img)
                    # image = image.convert('L')
                    # image.save(os.path.join(store_dir, '%d.jpeg'  % (i + 1)))
                    cv2.imwrite(os.path.join(store_dir, '%d.jpeg' % (i + 1)), img)
                    cv2.imwrite(os.path.join(store_tiff_dir, '%d.tiff' % (i + 1)), img_tiff)

                if len(os.listdir(store_dir)) == 0:
                    os.removedirs(store_dir)
            else:
                return

            new_patient_info['ImageDIR'] = store_dir
            new_patient_info['TiffDIR'] = store_tiff_dir
            new_patient_info['ConvertImage'] = convert_image_path
            if patient_id in self.patients_info:
                self.patients_info[patient_id].append(new_patient_info)
            else:
                self.patients_info[patient_id] = [new_patient_info,]
            self.sop_ids.append(uniqueid)
        except Exception as e:
            # traceback.print_exc()
            if '.jp' not in file_path or '.tif' not in file_path:
                print(file_path, repr(e))
            try:
                if len(os.listdir(store_dir)) == 0:
                    os.removedirs(store_dir)
                if len(os.listdir(store_tiff_dir)) == 0:
                    os.removedirs(store_tiff_dir)
            except Exception:
                return

    def reformat_file(self, root_paths):
        for root_path in root_paths:
            print(root_path)
            for (root, dir, files) in os.walk(root_path):
                for file in files:
                    self.read_file(os.path.join(root, file))

    def save_info(self):
        sop_ids_json = json.dumps(self.sop_ids).encode("utf-8")
        patient_info_json = json.dumps(self.patients_info).encode("utf-8")
        info_path = self.result + str(len(os.listdir(self.result))+1)
        os.makedirs(info_path)
        with open(os.path.join(info_path, 'Sop_ids.txt'), 'wb') as f:
            f.write(sop_ids_json)
            f.close()
        with open(os.path.join(info_path, 'Sop_ids_bak.txt'), 'wb') as f:
            f.write(sop_ids_json)
            f.close()
        with open(os.path.join(info_path, 'patients_info.txt'), 'wb') as f:
            f.write(patient_info_json)
            f.close()
        with open(os.path.join(info_path, 'patients_info_bak.txt'), 'wb') as f:
            f.write(patient_info_json)
            f.close()


if __name__ == '__main__':
    import sys
    from datetime import datetime
    # print(os.path.curdir)
    root_path, result_path, result_json = sys.argv[1:4]
    nn = moya_diag()
    start_time = datetime.now()
    info_path = result_json + str(len(os.listdir(result_json)))
    patient_info_path = os.path.join(info_path, 'patients_info.txt')
    sop_ids_path = os.path.join(info_path, 'Sop_ids.txt')
    patient_info = {}
    sop_ids = []

    if os.path.exists(patient_info_path):
        print('load patient info from %s' % patient_info_path)
        f = open(patient_info_path, encoding='utf-8')
        patient_info = json.load(f)
    if os.path.exists(sop_ids_path):
        print('load sop_ids info from %s' % sop_ids_path)
        f = open(sop_ids_path, encoding='utf-8')
        sop_ids = json.load(f)

    rdf = ReformatDsaFile(root_path, result_path, result_json, (750, 750),  sop_ids=sop_ids, patients_info=patient_info)
    # rdf.read_file('../IM000001')

    rdf.reformat_file(root_path.split(','))
    rdf.save_info()
    end_time = datetime.now()
    print((end_time-start_time).seconds)
