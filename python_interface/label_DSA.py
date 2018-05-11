# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets, QtGui

from pyqt.bak.labelDSA2 import Ui_MainWindow

import os

import json

class DSA_Label_Window(QtWidgets.QWidget, Ui_MainWindow):

    def __init__(self):
        super(DSA_Label_Window, self).__init__()
        self.setupUi(self)
        self.pwd = './normal_moyamoya_img'
        self.json_dict = {}
        self.angle = 0.0
        self.position = ''
        self.image_idx = 0
        self.image_folder = ''
        self.image_address = ''
        self.patient_name_list = []
        self.patient_id = ''
        self.patient_idx = 0
        self.patient_dsa_idx = 0
        self.patient_dsa_num = 0
        self.image_idx = 0
        self.dsa_idx = 1
        self.progress_value = 0
        self.dsa_total = 0

        self.load_Button.clicked.connect(self.load_data)
        self.lastImg_Button.clicked.connect(self.last_dsa)
        self.accept_Button.clicked.connect(self.next_dsa)
        self.dump_json_Button.clicked.connect(self.dump_json)
        self.set_LICA_Button.clicked.connect(self.LICA_state)
        self.set_RICA_Button.clicked.connect(self.RICA_state)
        self.set_LVA_Button.clicked.connect(self.LVA_state)
        self.set_RVA_Button.clicked.connect(self.RVA_state)
        self.set_LECA_Button.clicked.connect(self.LECA_state)
        self.set_RECA_Button.clicked.connect(self.RECA_state)
        self.set_LCCA_Button.clicked.connect(self.LCCA_state)
        self.set_RCCA_Button.clicked.connect(self.RCCA_state)
        self.set_CCAp_Button.clicked.connect(self.CCAp_state)
        self.set_CCAd_Button.clicked.connect(self.CCAd_state)
        self.set_LSUBA_Button.clicked.connect(self.LSUBA_state)
        self.set_RSUBA_Button.clicked.connect(self.RSUBA_state)
        self.set_ARCH_Button.clicked.connect(self.ARCH_state)


        self.set_LAT_ICA_Button.clicked.connect(self.ICA_state)
        self.set_LAT_VA_Button.clicked.connect(self.VA_state)
        self.set_LAT_ECA_Button.clicked.connect(self.ECA_state)
        self.set_UnknownAP_Button.clicked.connect(self.UnknownAP)
        self.set_UnknownLAT_Button.clicked.connect(self.UnknownLAT)
        self.remove_Button.clicked.connect(self.Remove_Button)

        self.verticalSlider.valueChanged.connect(self.slider_play)


    def LICA_state(self):
        self.set_dsa_type('LICA')

    def RICA_state(self):
        self.set_dsa_type('RICA')

    def LVA_state(self):
        self.set_dsa_type('LVA')

    def RVA_state(self):
        self.set_dsa_type('RVA')

    def LECA_state(self):
        self.set_dsa_type('LECA')

    def RECA_state(self):
        self.set_dsa_type('RECA')

    def LCCA_state(self):
        self.set_dsa_type('LCCA')

    def RCCA_state(self):
        self.set_dsa_type('RCCA')

    def CCAp_state(self):
        self.set_dsa_type('CCAp')

    def CCAd_state(self):
        self.set_dsa_type('CCAd')

    def LSUBA_state(self):
        self.set_dsa_type('LSUBA')

    def RSUBA_state(self):
        self.set_dsa_type('RSUBA')

    def ARCH_state(self):
        self.set_dsa_type('ARCH')

    def ICA_state(self):
        self.set_dsa_type('LAT_ICA')

    def VA_state(self):
        self.set_dsa_type('LAT_VA')

    def ECA_state(self):
        self.set_dsa_type('LAT_ECA')

    def UnknownAP(self):
        self.set_dsa_type('Unknown_AP')

    def UnknownLAT(self):
        self.set_dsa_type('Unknown_LAT')

    def Remove_Button(self):
        self.set_dsa_type('')

    def slider_play(self):
        pos = self.verticalSlider.value()
        frame = 100/(self.json_dict[self.patient_id][self.patient_dsa_idx]['NumberOfFrames'])
        self.image_idx = round(pos/frame)
        self.display_image()

    def update_progress(self):
        self.progress_value = int(self.dsa_idx / self.dsa_total * 99)
        self.progress_Bar.setValue(self.progress_value)

    def dump_json(self):
        json_name = self.get_save_filename()
        with open(os.path.join(self.pwd, json_name), 'w') as jf:
            json.dump(self.json_dict, jf, indent=1)

    def last_dsa(self):
        if self.patient_dsa_idx != 0:
            self.patient_dsa_idx -= 1
            self.dsa_idx -= 1
        elif self.patient_idx != 0:
            self.patient_idx -= 1
            self.patient_id = self.patient_name_list[self.patient_idx]
            self.patient_dsa_idx = len(self.json_dict[self.patient_id]) - 1
            self.dsa_idx -= 1
        self.update_progress()
        self.display_image()
        self.display_info()
        self.set_image_idx()

    def set_dsa_type(self, dsa_type):
        self.json_dict[self.patient_id][self.patient_dsa_idx]['ArteryType'] = dsa_type
        self.next_dsa()

    def next_dsa(self):
        if (self.patient_dsa_idx + 1) < len(self.json_dict[self.patient_id]):
            self.patient_dsa_idx += 1
            self.dsa_idx += 1
        elif (self.patient_idx + 1) < len(self.patient_name_list):
            self.patient_idx += 1
            self.patient_id = self.patient_name_list[self.patient_idx]
            self.patient_dsa_idx = 0
            self.dsa_idx += 1
        self.update_progress()
        print(self.patient_id, self.patient_dsa_idx)
        self.display_image()
        self.display_info()
        self.set_image_idx()

    def count_dsa(self,json_dict):
        dsa_total = 0
        for key in json_dict:
            dsa_total += len(json_dict[key])
        return dsa_total

    def load_data(self):
        json_file = self.setOpenFileName()
        with open(json_file, 'r', encoding='utf-8') as f:
            self.json_dict = json.loads(f.read())
        for key in self.json_dict:
            self.patient_name_list.append(key)
        self.patient_id = self.patient_name_list[self.patient_idx]
        self.dsa_total = self.count_dsa(self.json_dict)
        self.update_progress()
        self.set_image_idx()
        self.display_image()
        self.display_info()

    def set_image_idx(self):
        self.image_idx = int(self.json_dict[self.patient_id][self.patient_dsa_idx]['NumberOfFrames'] / 2)

    def get_image_address(self):
        image_folder = self.json_dict[self.patient_id][self.patient_dsa_idx]['ImageDIR']
        image_address = self.pwd + '/' + image_folder + '/' + str(self.image_idx) + '.jpg'
        return image_address

    def display_image(self):
        self.image_address = self.get_image_address()

        # dsa_image = cv2.imread(self.image_address, cv2.IMREAD_GRAYSCALE)
        # print(dsa_image)
        # dsa_image = cv2.resize(dsa_image,(750,750))
        # dsa_image = dsa_image.astype(np.uint8)
        # qimage = QtGui.QImage(dsa_image.tostring(), dsa_image.shape[1], dsa_image.shape[0],
        #                       QtGui.QImage.Format_Grayscale8)
        pic = QtGui.QPixmap(self.image_address).scaled(750, 750)
        self.imageLabel.setPixmap(pic)

    def display_info(self):
        if 'Position' in self.json_dict[self.patient_id][self.patient_dsa_idx]:
            if 'ArteryType' in self.json_dict[self.patient_id][self.patient_dsa_idx]:
                self.PrimaryAngel_label.setText(str(self.json_dict[self.patient_id][self.patient_dsa_idx]['PositionerPrimaryAngle']) +
                                                '\n ' + self.json_dict[self.patient_id][self.patient_dsa_idx]['Position'] +
                                                '\n ' + self.json_dict[self.patient_id][self.patient_dsa_idx]['ArteryType'] +
                                                '\n' + str(self.dsa_idx) + '/' + str(self.dsa_total))
            else:
                self.PrimaryAngel_label.setText(
                    str(self.json_dict[self.patient_id][self.patient_dsa_idx]['PositionerPrimaryAngle']) +
                    '\n ' + self.json_dict[self.patient_id][self.patient_dsa_idx]['Position'] +
                    '\n' + str(self.dsa_idx) + '/' + str(self.dsa_total))
        else:
            self.PrimaryAngel_label.setText(
                str(self.json_dict[self.patient_id][self.patient_dsa_idx]['PositionerPrimaryAngle']))


    def setOpenFileName(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                   "OpenDicomFile",
                                                                   self.pwd,
                                                                   'All Files (*);; JSON Files (*.json)')
        return fileName

    def get_save_filename(self):
        filename, filetype = QtWidgets.QFileDialog.getSaveFileName(self,
                                                                   "SaveJsonFile",
                                                                   self.pwd,
                                                                   "All Files (*);; JSON Files (*.json)")
        return filename







if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    dsa_label_App = DSA_Label_Window()
    dsa_label_App.show()
    sys.exit(app.exec_())