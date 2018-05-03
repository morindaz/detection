# -*- coding: utf-8 -*-
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from skimage import img_as_ubyte
from PIL import Image


class ArteryNN(nn.Module):
    def __init__(self):
        super(ArteryNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,8,3,1,0),   # input_channel, output_channel, kernel size, stride size, padding
            nn.Conv2d(8,8,3,1,0),
            nn.Conv2d(8,8,3,1,1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride= 2))#(1,8,98,98)
        # conv1输出为(16, 98, 98)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8,24 , 3,1,0),
            nn.Conv2d(24,24,3,1,1),# 96 96
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2,2) #32 64 48 48
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(24,48,3,1,1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2,2)#64,128,24,24
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(48,96,3,1,1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2,2) # 256,12,12
        )
        # conv2输出为(128 , 24, 24)
        self.fc1 = nn.Sequential(
            nn.Linear(96*12*12,1024),
            nn.Dropout(0.5),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(1024,128),
            nn.Dropout(0.5),
            nn.ReLU())
        self.output = nn.Sequential(
            nn.Linear(128, 6))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        prediction = self.output(x)
        return prediction

class moya_ica(nn.Module):
    def __init__(self):
        super(moya_ica, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride= 2))#(1,8,100,120)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 24, 3, 1, 1),
            nn.Conv2d(24, 24, 3, 1, 1),# 96 96
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) #8 24 50 60
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(24, 48, 3, 1, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)#24,48,25,30
        )
        self.fc1 = nn.Sequential(
            nn.Linear(48*25*30, 1024),
            nn.Dropout(0.5),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Dropout(0.5),
            nn.ReLU())
        self.output = nn.Sequential(
            nn.Linear(128, 3))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        prediction = self.output(x)
        return prediction
# CUDA = False
# if torch.cuda.is_available():
#     CUDA = True
#
# if CUDA:
#     artery_cnn_sp1 = ArteryNN().cuda()
#     artery_cnn_sp2 = ArteryNN().cuda()
#     artery_cnn_sp3 = ArteryNN().cuda()
# else:


imagesize = (200, 200)

class moya_diag():
    def __init__(self):
        self.artery_cnn_sp3 = ArteryNN()
        self.artery_cnn_sp2 = ArteryNN()
        self.artery_cnn_sp1 = ArteryNN()

        self.artery_cnn_sp1.load_state_dict(torch.load('sp1_params.pkl', map_location=lambda storage, loc: storage))
        self.artery_cnn_sp2.load_state_dict(torch.load('sp2_params.pkl', map_location=lambda storage, loc: storage))
        self.artery_cnn_sp3.load_state_dict(torch.load('sp3_params.pkl', map_location=lambda storage, loc: storage))

        # self.icaStatusCNN = moya_ica()
        # self.icaStatusCNN.load_state_dict(torch.load('lica_.pkl', map_location=lambda storage, loc: storage))
        # self.acaStatusCNN = moya_ica()
        # self.acaStatusCNN.load_state_dict(torch.load('laca_.pkl', map_location=lambda storage, loc: storage))
        # self.mcaStatusCNN = moya_ica()
        # self.mcaStatusCNN.load_state_dict(torch.load('lmca_.pkl', map_location=lambda storage, loc: storage))
        # self.moyaStatusCNN = moya_ica()
        # self.moyaStatusCNN.load_state_dict(torch.load('lmoya.pkl', map_location=lambda storage, loc: storage))

    def Min_Max_Scaling(self,image):
        min = np.min(image)
        max = np.max(image)
        output = (image-min)/(max-min)
        return output

    def convertLabel(self,label):
        dsalabel_dict = {'LICA': 1, 'RICA': 2, 'LVA': 3, 'RVA': 4, 'LECA': 5, 'RECA': 0,
                         1: 'LICA', 2: 'RICA', 3: 'LVA', 4: 'RVA', 5: 'LECA', 0: 'RECA'}
        return dsalabel_dict.get(label)

    def statusLabel(self,label, type):
        if type == 'moya':
            statusDict = {0: 'Normal', 1: 'Moyamoya'}
        else:
            statusDict = {0: 'Normal', 1: 'Stenosis', 2: 'Occlusion'}
        return statusDict.get(label)


    def predDicom(self, frameArray, manufacturer):
        img_num,img_w,img_h = frameArray.shape
        if manufacturer == 'GE MEDICAL SYSTEMS':
            start_point = int(np.ceil(img_num*0.2))
            end_point = int(np.ceil(img_num*0.7))
        else:
            start_point = int(np.ceil(img_num*0.25))
            end_point = int(np.ceil(img_num*0.6))


        sp1 = np.zeros((img_w,img_h), dtype=np.float32)
        sp2 = np.zeros((img_w,img_h), dtype=np.float32)

        for m in range(start_point, end_point):
            image = frameArray[m].astype(np.float32)
            sp1 = cv2.add(sp1, image)
        for n in range(img_num):
            image = frameArray[n].astype(np.float32)
            sp2 = cv2.add(sp2, image)

        sp1 = sp1/(end_point-start_point)
        sp2 = sp2/img_num
        sp3 = sp2 - (frameArray[0] + frameArray[-1])/2


        super1 = cv2.resize(img_as_ubyte(self.Min_Max_Scaling(sp1)), imagesize)
        super2 = cv2.resize(img_as_ubyte(self.Min_Max_Scaling(sp2)), imagesize)
        super3 = cv2.resize(img_as_ubyte(self.Min_Max_Scaling(sp3)), imagesize)

        sp1_output = self.net_pred(super1, self.artery_cnn_sp1)
        sp2_output = self.net_pred(super2, self.artery_cnn_sp2)
        sp3_output = self.net_pred(super3, self.artery_cnn_sp3)

        if sp1_output == sp2_output:
            prediction = sp1_output
        elif sp1_output == sp3_output:
            prediction = sp1_output
        elif sp2_output == sp3_output:
            prediction = sp2_output
        else:
            prediction = sp1_output
        return prediction

    def net_pred(self, image, net, dsatype=None):
        net.eval()
        if dsatype:
            image = image.reshape(-1, 1, 200, 240)
        else:
            image = image.reshape(-1, 1, imagesize[1], imagesize[0])

        test_data = torch.from_numpy(image).float()
        test_data = Variable(test_data, volatile=False)

        output = net(test_data)

        pred = torch.max(output, 1)[1].data
        pred = pred.numpy()
        pred_out = pred[0]
        if dsatype:
            return self.statusLabel(pred_out, dsatype)
        else:
            return self.convertLabel(pred_out)

    def predMoyaStatus(self, roi):
        icaStatus_label = self.net_pred(roi, self.icaStatusCNN, 'ica')
        acaStatus_label = self.net_pred(roi, self.acaStatusCNN, 'aca')
        mcaStatus_label = self.net_pred(roi, self.mcaStatusCNN, 'mca')
        moyaStatus_label = self.net_pred(roi, self.moyaStatusCNN, 'moya')
        statusList = {'ICA Status': icaStatus_label, 'ACA Status': acaStatus_label,
                      'MCA Status': mcaStatus_label, 'Moyamoya Status': moyaStatus_label}
        return statusList










