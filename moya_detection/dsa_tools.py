
# -*- coding: utf-8 -*-
__author__ = 'Zhonghua Hu'
__version__ = 1.0



import copy
from pathlib import Path
import numpy as np
from skimage import img_as_ubyte
import os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import SimpleITK as sitk
import numpy as np
import json
import shutil
import cv2


def ROI(mask_size, image_address, bbox_size, stride):
    mask = np.ones(mask_size)
    image = cv2.imread(image_address, cv2.IMREAD_GRAYSCALE)
    print(image_address)
    image_size = image.shape
    filter_dict = {}
    x_bound_start = int(image_size[1]*0.5)
    x_bound_end = int(image_size[1]*0.8)
    y_bound_start = int(image_size[0]*0.3)
    y_bound_end = int(image_size[0]*0.7)
    for x_idx in range(x_bound_start, x_bound_end,stride):
        for y_idx in range(y_bound_start, y_bound_end ,stride):
            conv_value = np.sum(np.multiply(mask, image[x_idx: x_idx + mask_size[1], y_idx: y_idx + mask_size[0]]))
            filter_dict[x_idx, y_idx] = conv_value
    min_x, min_y = min(filter_dict, key=filter_dict.get)

    if min_y < round(image_size[0]/2):
        top_left_vertex = (max(min_y - int(0.5 * bbox_size[0]),0), min_x -int(0.8*bbox_size[1]))
        bottom_right_vertex = (min_y + int(0.5 * bbox_size[0]), min_x + int(0.2 * bbox_size[1]))

        crop_image = image[top_left_vertex[1]:bottom_right_vertex[1], top_left_vertex[0]:bottom_right_vertex[0]]
        cv2.rectangle(image, top_left_vertex, bottom_right_vertex, (0, 255, 0), 1)
    else:
        top_left_vertex = (min_y - int(0.5*bbox_size[0]), min_x - int(0.8*bbox_size[1]))
        bottom_right_vertex = (min(min_y + int(0.5 * bbox_size[0]), image_size[1]-1), min_x + int(0.2 * bbox_size[1]))

        crop_image = image[top_left_vertex[1]:bottom_right_vertex[1], top_left_vertex[0]:bottom_right_vertex[0]]
        cv2.rectangle(image, top_left_vertex, bottom_right_vertex, (0, 255, 0), 1)
    #cv2.rectangle(image, (min_y, min_x), (min_y + mask_size[1], min_x + mask_size[0]), (0, 255, 0), 3)
    return image, crop_image

def ROI2(mask_size, image, bbox_size, stride):
    mask = np.ones(mask_size)
    image_size = image.shape
    filter_dict = {}
    x_bound_start = int(image_size[1]*0.5)
    x_bound_end = int(image_size[1]*0.8)
    y_bound_start = int(image_size[0]*0.3)
    y_bound_end = int(image_size[0]*0.7)
    for x_idx in range(x_bound_start, x_bound_end,stride):
        for y_idx in range(y_bound_start, y_bound_end ,stride):
            conv_value = np.sum(np.multiply(mask, image[x_idx: x_idx + mask_size[1], y_idx: y_idx + mask_size[0]]))
            filter_dict[x_idx, y_idx] = conv_value
    min_x, min_y = min(filter_dict, key=filter_dict.get)

    if min_y < round(image_size[0]/2):
        top_left_vertex = (max(min_y - int(0.5 * bbox_size[0]),0), min_x -int(0.8*bbox_size[1]))
        bottom_right_vertex = (min_y + int(0.5 * bbox_size[0]), min_x + int(0.2 * bbox_size[1]))
        cv2.rectangle(image, top_left_vertex, bottom_right_vertex, (0, 255, 0), 1)
        crop_image = image[top_left_vertex[1]:bottom_right_vertex[1], top_left_vertex[0]:bottom_right_vertex[0]]
    else:
        top_left_vertex = (min_y - int(0.5*bbox_size[0]), min_x - int(0.8*bbox_size[1]))
        bottom_right_vertex = (min(min_y + int(0.5 * bbox_size[0]), image_size[1]-1), min_x + int(0.2 * bbox_size[1]))
        cv2.rectangle(image, top_left_vertex, bottom_right_vertex, (0, 255, 0), 1)
        crop_image = image[top_left_vertex[1]:bottom_right_vertex[1], top_left_vertex[0]:bottom_right_vertex[0]]
    return image, crop_image

def get_label(json):
    arteryType = json['ArteryType']
    moyamoya_status = json['MYMYStatus']
    return arteryType, moyamoya_status

def Min_Max_Scaling(image):
    min_gray = np.min(image)
    max_gray = np.max(image)
    output = (image - min_gray) / (max_gray - min_gray)
    return output

def load_json(jsonfile):
    f = open(jsonfile, "r", encoding="utf-8")
    json_file = json.loads(f.read())
    f.close()
    return json_file

def load_info(filename):
    import dicom
    info = {}
    try:
        dicom_info = dicom.read_file(filename)
        info['ImageType'] = dicom_info.ImageType[0]  # 'ORIGINAL' or 'DERIVED':
        info['PatientID'] = dicom_info.PatientID
        info['PatientName'] = str(dicom_info.PatientName)
        info['PatientBirthDate'] = dicom_info.PatientBirthDate
        info['PatientSex'] = dicom_info.PatientSex
        info['StudyID'] = dicom_info.StudyID
        info['StudyDate'] = dicom_info.StudyDate
        info['StudyTime'] = dicom_info.StudyTime
        info['InstitutionName'] = dicom_info.InstitutionName
        info['Manufacturer'] = dicom_info.Manufacturer  # GE or Philips
        info['NumberOfFrames'] = int(dicom_info.NumberOfFrames)
        info['SOPInstanceUID'] = str(dicom_info.SOPInstanceUID)
        info['InstanceNumber'] = str(dicom_info.InstanceNumber)
        info['PositionerPrimaryAngle'] = float(dicom_info.PositionerPrimaryAngle)  # use to detect position
        if info['Manufacturer'] == 'GE MEDICAL SYSTEMS':
            img_dir = os.path.join(info['PatientID'], info['SOPInstanceUID'])
            if 30.0 >= info['PositionerPrimaryAngle'] >= -30.0:
                info['Position'] = 'AP'  # AP 正位
            elif 120 >= info['PositionerPrimaryAngle'] >= 60:
                info['Position'] = 'LAT'  # LAT 侧位
        else:
            img_dir = os.path.join(info['PatientID'], info['SOPInstanceUID'] + '.' + info['InstanceNumber'])
            if 15.0 >= info['PositionerPrimaryAngle'] >= -15.0:
                info['Position'] = 'AP'
            elif 105 >= info['PositionerPrimaryAngle'] >= 75:
                info['Position'] = 'LAT'

        info['PositionerSecondaryAngle'] = float(dicom_info.PositionerSecondaryAngle)  # 作用还不了解
        info['ImageDIR'] = img_dir
    except Exception:
        return
    else:
        return info

def export_images(filename):
    try:
        ds = sitk.ReadImage(filename)
        img_array = sitk.GetArrayFromImage(ds)
        frame_num, width, height = img_array.shape
    except Exception:
        return 'error'
    else:
        return img_array, frame_num, width, height

def get_DSA(dsa_folder, tar_dsa, tar_img):
    infos = {}
    dsa_number = 0
    IMGDIR_list = []
    all_files = os.walk(dsa_folder)
    for file in all_files:
        if tar_dsa in file[0] or tar_img in file[0]:
            continue
        if (len(file[2]) > 0) and (len(file[1]) == 0):
            for filename in file[2]:
                file_info = load_info(os.path.join(file[0], filename))
                if isinstance(file_info, dict):
                    print(file[0], file_info['ImageType'], file_info['ImageDIR'],
                          file_info['NumberOfFrames'])  # DSA类型，分类目录，帧数
                    if file_info['ImageType'] == 'DERIVED' and 50 >= file_info['NumberOfFrames'] >= 5 and (
                            file_info['ImageDIR'] not in IMGDIR_list):  # 筛选去除 剪影 旋转 路途 和相同的DSA
                        # copy 合规DSA到DSA 文件夹，按patientID分类
                        IMGDIR_list.append(file_info['ImageDIR'])
                        dsa_number += 1
                        print(dsa_number)
                        file_image = export_images(os.path.join(file[0], filename))
                        if file_info['PatientID'] not in infos:
                            infos[file_info['PatientID']] = [file_info]
                        else:
                            infos[file_info['PatientID']].append(file_info)
                        isExist1 = os.path.exists(os.path.join(tar_dsa, file_info['PatientID']))
                        if not isExist1:
                            os.mkdir(os.path.join(tar_dsa, file_info['PatientID']))
                        shutil.copyfile(os.path.join(file[0], filename),
                                        os.path.join(tar_dsa, file_info['PatientID'],
                                                     file_info['SOPInstanceUID'] + '.dcm')
                                        )

                        # extract img from DSA 提取DSA的分帧图片，按patientID和dsa分类
                        isExist2 = os.path.exists(os.path.join(tar_img, file_info['PatientID']))
                        if not isExist2:
                            os.mkdir(os.path.join(tar_img, file_info['PatientID']))
                        img_dir = file_info['ImageDIR']
                        isExist3 = os.path.exists(os.path.join(tar_img, img_dir))
                        if not isExist3:
                            os.mkdir(os.path.join(tar_img, img_dir))
                        if file_image != 'error':
                            for i in range(file_image[1]):
                                cv2.imwrite(os.path.join(tar_img, img_dir, str(i) + '.jpg'),
                                            file_image[0][i].astype(float) * 255 / (np.max(file_image[0][i])))

    with open(os.path.join(tar_img, 'info.json'), 'w') as file:  # dump infos into json file
        json.dump(infos, file, indent=1)
    file.close()

def SaveImage(image, image_name, imagefolder):
    if not os.path.exists(imagefolder):
        os.mkdir(imagefolder)
    image = img_as_ubyte(Min_Max_Scaling(image))
    cv2.imwrite(os.path.join(imagefolder, image_name), image)

def SaveLabel(info, item, label_name,labelfolder):
    if not os.path.exists(labelfolder):
        os.mkdir(labelfolder)
    labellist = []
    f = open(info, 'r', encoding='utf-8')
    infojson = json.loads(f.read())
    f.close()
    for num, patID in enumerate(infojson):
        for i in range(len(infojson[patID])):
            pat_dsa = infojson[patID][i]
            labellist.append([(pat_dsa['SOPInstanceUID'] +'.jpg'), pat_dsa['ArteryType'], pat_dsa['MoyamoyaStatus'][item]])
    with open(os.path.join(labelfolder, label_name), 'w') as flabel:
        for line in labellist:
            flabel.write(str(line) + '\n')
    flabel.close()

def load_Img(imgFolderName):  # 提取某文件夹下所有图片
    imgs = os.listdir(imgFolderName)
    imgNum = len(imgs)
    for i in range(imgNum):
        imgs[i] = imgs[i].split('.')
        imgs[i][0] = int(imgs[i][0])
    imgs.sort()
    for i in range(imgNum):
        imgs[i][0] = str(imgs[i][0])
        imgs[i] = imgs[i][0] + '.' + imgs[i][1]
    imgsample = cv2.imread(os.path.join(imgFolderName, imgs[0]), cv2.IMREAD_GRAYSCALE)
    imageshape = imgsample.shape
    img = np.empty((imgNum, imageshape[0], imageshape[1]), dtype=np.float32)
    for i in range(imgNum):
        img[i] = cv2.imread(os.path.join(imgFolderName, imgs[i]), cv2.IMREAD_GRAYSCALE)
    return img, imgs

# def arteryStatusLabel(imagename,imagefolder):
#     image = cv2.imread(os.path.join(imagefolder, imagename),cv2.IMREAD_GRAYSCALE)
#     cv2.imshow(imagename, image)

def convertDSALabel(label):
    dsalabel_dict = {'LICA' : 1, 'RICA' : 2, 'LVA' : 3,
                     'RVA': 4, 'LECA': 5, 'RECA': 0}
    return np.int(dsalabel_dict[label])

def GetID(jsonfile):
    pID_list = []
    keys = jsonfile.keys()
    for num, key in enumerate(keys):
        pID_list.append(key)
    return pID_list

def JsonRmDup(inputjson, outputjson, outputfolder):
    f = open(inputjson, 'r', encoding='utf-8')
    infos = json.loads(f.read())
    f.close()
    for num, patientID in enumerate(infos.keys()):
        print(patientID)
        temp = []
        for item in infos[patientID]:
            if item not in temp:
                temp.append(item)
        infos[patientID] = temp
    with open(os.path.join(outputfolder,outputjson), 'w') as file:
        json.dump(infos, file, indent=1)
    file.close()

def flip_dsa_label(ori_dsa):
    flip_map = {
        'LICA': 'RICA',
        'LVA': 'RVA',
        'LECA': 'RECA',
        'RICA': 'LICA',
        'RVA': 'LVA',
        'RECA': 'LECA',
        0 : 1,
        1 : 0,
    }
    return flip_map.get(ori_dsa)

def super_class(imagefolder, device_type, resize):
    images, files = load_Img(imagefolder)
    [img_num, img_w, img_h] = images.shape
    if device_type == 'GE MEDICAL SYSTEMS':
        start_point = int(np.ceil(img_num * 0.2))
        end_point = int(np.ceil(img_num * 0.7))
    else:
        start_point = int(np.ceil(img_num * 0.25))
        end_point = int(np.ceil(img_num * 0.6))
    sp1 = np.zeros((img_w, img_h), dtype=np.float32)
    sp2 = np.zeros((img_w, img_h), dtype=np.float32)
    for m in range(start_point, end_point):
        sp1 = cv2.add(sp1, images[m])
    for n in range(img_num):
        super2 = cv2.add(sp2, images[n])
    sp1 = img_as_ubyte(Min_Max_Scaling(cv2.resize(sp1 / (end_point - start_point), resize)))
    sp2 = img_as_ubyte(Min_Max_Scaling(cv2.resize(super2 / (img_num), resize)))
    sp3 = img_as_ubyte(Min_Max_Scaling(cv2.resize(super2 - (images[0] + images[-1]) / 2, resize)))
    return sp1, sp2, sp3

def super_diag(imagefolder, device_type, resize):
    images, files = load_Img(imagefolder)
    [img_num, img_w, img_h] = images.shape
    if device_type == 'GE MEDICAL SYSTEMS':
        mask_start = 0
        mask_end = int(np.floor(img_num * 0.15))
        start_point = mask_end + 1
        if start_point + 6 <= img_num - 1:
            end_point = start_point + 6
        else:
            end_point = img_num - 1
    else:
        mask_start = 1
        mask_end = int(np.ceil(img_num * 0.15))
        start_point = mask_end + 1
        end_point = int(np.ceil(img_num * 0.5))
    supermask = np.zeros((img_w, img_h), dtype=np.float32)
    superimpose1 = np.zeros((img_w, img_h), dtype=np.float32)
    for mn in range(mask_start, mask_end):
        supermask = cv2.add(supermask, images[mn])

    for m in range(start_point, end_point):
        superimpose1 = cv2.add(superimpose1, images[m])
    #supermask = np.log(supermask / (mask_end - mask_start)+0.001)
    supermask = supermask/(mask_end - mask_start)
    #superimpose1 = np.log(superimpose1 / (end_point - start_point)+0.001)
    superimpose1 = superimpose1/(end_point - start_point)
    superimpose = superimpose1 - supermask
    minsuper1 = np.min(superimpose)
    if minsuper1 < 0:
        #superimpose = np.exp(superimpose - minsuper1)
        superimpose = superimpose - minsuper1
    super = cv2.resize(img_as_ubyte(Min_Max_Scaling(superimpose)), resize)
    return super

def super_diag2(images,device_type, resize):

    [img_num, img_w, img_h] = images.shape
    if device_type == 'GE MEDICAL SYSTEMS':
        mask_start = 0
        mask_end = int(np.floor(img_num * 0.15))
        start_point = mask_end + 1
        if start_point + 6 <= img_num - 1:
            end_point = start_point + 6
        else:
            end_point = img_num - 1
    else:
        mask_start = 1
        mask_end = int(np.ceil(img_num * 0.15))
        start_point = mask_end + 1
        end_point = int(np.ceil(img_num * 0.5))
    supermask = np.zeros((img_w, img_h), dtype=np.float32)
    superimpose1 = np.zeros((img_w, img_h), dtype=np.float32)
    for mn in range(mask_start, mask_end):
        supermask = cv2.add(supermask, images[mn].astype(np.float32))
    for m in range(start_point, end_point):
        superimpose1 = cv2.add(superimpose1, images[m].astype(np.float32))
    #supermask = np.log(supermask / (mask_end - mask_start)+0.001)
    supermask = supermask/(mask_end - mask_start)
    #superimpose1 = np.log(superimpose1 / (end_point - start_point)+0.001)
    superimpose1 = superimpose1/(end_point - start_point)
    superimpose = superimpose1 - supermask
    minsuper1 = np.min(superimpose)
    if minsuper1 < 0:
        #superimpose = np.exp(superimpose - minsuper1)
        superimpose = superimpose - minsuper1
    super = cv2.resize(img_as_ubyte(Min_Max_Scaling(superimpose)), resize)
    return super

def get_ica(jsonfile, output,jsonoutput):
    if not os.path.exists(output):
        os.mkdir(output)
    else:
        shutil.rmtree(output)
        os.mkdir(output)
    info = load_json(jsonfile)
    temp = copy.deepcopy(info)
    pidlist = GetID(info)
    ICA_num = 0
    for i in range(len(pidlist)):
        patID = pidlist[i]
        dellist =[]
        for ii in range(len(info[patID])):
            if info[patID][ii]['ArteryType'] in ['LICA', 'RICA']:
                ICA_num += 1
                imagename = info[patID][ii]['ImageDIR']
                shutil.copytree(imagename, os.path.join(output, imagename))
                print(imagename, ICA_num)
            else:
                dellist.append(temp[patID][ii]['SOPInstanceUID'])
        if dellist != []:
            for item in info[patID]:
                if item['SOPInstanceUID'] in dellist:
                    temp[patID].remove(item)
        if temp[patID] == []:
            temp.pop(patID)
    with open(os.path.join(output, jsonoutput), 'w') as jsonf:
        json.dump(temp, jsonf, indent=1)
    jsonf.close()

def label_Artery(image):
    key_Artery = {48: 'UNKNOWN', 49: 'LICA', 50: 'RICA', 51: 'LVA',
                  52: 'RVA', 53: 'LECA', 54: 'RECA', 55: 'LCCA', 56: 'RCCA'}
    cv2.imshow('IMAGE', image)
    press_key = cv2.waitKey(-1)
    label = key_Artery[press_key]
    cv2.destroyAllWindows()
    return label



# def label_tools(imagename):
#     cv2.imread(imagename)



#def img_augument(img, flip=True)







