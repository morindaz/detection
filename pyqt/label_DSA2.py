# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets, QtGui

from artery_label.labelDSA2 import Ui_MainWindow

import os

import json

class DSA_Label_Window(QtWidgets.QWidget, Ui_MainWindow):

    def __init__(self):
        super(DSA_Label_Window, self).__init__()
        self.setupUi(self)
        self.pwd = 'E:/intern/test/'
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
        self.image_idx = 1
        self.dsa_idx = 1
        self.progress_value = 0
        self.dsa_total = 0
        self.image_dirs = []
        self.image_dir_idx = 0
        self.image_dir_label = {}

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
        numberOfFrames = len(os.listdir(self.image_dirs[self.image_dir_idx]))
        frame = 100/numberOfFrames
        self.image_idx = round(pos/frame)
        self.display_image()

    def update_progress(self):
        self.progress_value = int(self.image_dir_idx / self.dsa_total * 99)
        self.progress_Bar.setValue(self.progress_value)

    def dump_json(self):
        # json_name = self.get_save_filename()
        json_name = os.path.join(self.pwd, 'label.json')
        json_name_bak = os.path.join(self.pwd, 'label_bak.json')
        if os.path.exists(json_name):
            f = open(json_name, encoding='utf-8')
            labels = json.load(f)
            with open(os.path.join(self.pwd, json_name_bak), 'w') as jf:
                json.dump(labels, jf, indent=1)
            labels.update(self.image_dir_label)
            with open(os.path.join(self.pwd, json_name), 'w') as jf:
                json.dump(labels, jf, indent=1)
        else:
            with open(os.path.join(self.pwd, json_name), 'w') as jf:
                json.dump(self.image_dir_label, jf, indent=1)

    def last_dsa(self):
        if self.image_dir_idx != 0:
            self.image_dir_idx -= 1
            self.update_progress()
            self.set_image_idx()
            self.display_image()
            self.display_info()


    def set_dsa_type(self, dsa_type):
        self.image_dir_label[self.image_dirs[self.image_dir_idx]] = dsa_type

    def next_dsa(self):
        if len(self.image_dir_label) % 5 == 0:
            self.dump_json()
        if self.image_dir_idx < self.dsa_total-1:
            self.image_dir_idx += 1
            self.update_progress()
            # print(self.patient_id, self.patient_dsa_idx)
            self.set_image_idx()
            self.display_image()
            self.display_info()


    def load_data(self):
        for person_dir in os.listdir(self.pwd):
            real_person_dir = os.path.join(self.pwd, person_dir)
            if not os.path.isdir(real_person_dir):
                continue
            for dir in os.listdir(real_person_dir):
                if dir.startswith('LAT') or dir.startswith('AP'):
                    real_person_dsa_dir = os.path.join(real_person_dir, dir)
                    self.image_dirs.append(os.path.join(real_person_dsa_dir, 'all_image_1'))
        self.dsa_total = len(self.image_dirs)
        self.patient_dsaidx = 0
        self.dsa_idx = 0
        self.image_di_idx = 0
        self.update_progress()
        self.set_image_idx()
        self.display_image()
        self.display_info()

    def set_image_idx(self):
        numberOfFrames = len(os.listdir(self.image_dirs[self.image_dir_idx]))
        self.image_idx = int(numberOfFrames / 2)

    def get_image_address(self):
        # image_folder = self.json_dict[self.patient_id][self.patient_dsa_idx]['ImageDIR']
        # image_address = self.pwd + '/' + image_folder + '/' + str(self.image_idx) + '.jpg'
        image_address = os.path.join(self.image_dirs[self.image_dir_idx], str(self.image_idx) + '.jpeg')
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
        # print(self.image_address)
        self.imageLabel.setPixmap(pic)

    def display_info(self):
        image_path = self.image_dirs[self.image_dir_idx]
        if image_path in self.image_dir_label:
            self.PrimaryAngel_label.setText(image_path.split('\\')[-2] + '\n\nLabel:' + self.image_dir_label[image_path])
        else:
            self.PrimaryAngel_label.setText(image_path.split('\\')[-2])

        # if 'Position' in self.json_dict[self.patient_id][self.patient_dsa_idx]:
        #     if 'ArteryType' in self.json_dict[self.patient_id][self.patient_dsa_idx]:
        #         self.PrimaryAngel_label.setText(str(self.json_dict[self.patient_id][self.patient_dsa_idx]['PositionerPrimaryAngle']) +
        #                                         '\n ' + self.json_dict[self.patient_id][self.patient_dsa_idx]['Position'] +
        #                                         '\n ' + self.json_dict[self.patient_id][self.patient_dsa_idx]['ArteryType'] +
        #                                         '\n' + str(self.dsa_idx) + '/' + str(self.dsa_total))
        #     else:
        #         self.PrimaryAngel_label.setText(
        #             str(self.json_dict[self.patient_id][self.patient_dsa_idx]['PositionerPrimaryAngle']) +
        #             '\n ' + self.json_dict[self.patient_id][self.patient_dsa_idx]['Position'] +
        #             '\n' + str(self.dsa_idx) + '/' + str(self.dsa_total))
        # else:
        #     self.PrimaryAngel_label.setText(
        #         str(self.json_dict[self.patient_id][self.patient_dsa_idx]['PositionerPrimaryAngle']))


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