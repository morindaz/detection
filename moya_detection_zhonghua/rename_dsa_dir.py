#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import dicom
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')


class RenameDir(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.ignore_dirs = ['Aneurysm', 'Malformation', 'Moyamoya', 'Moyamoya Disease', 'Stenosis', 'Unknown']


    def read_file(self, file_path):
        try:
            dsa = dicom.read_file(file_path.encode('utf-8'), stop_before_pixels=True, force=True)
            try:
                study_day = dsa.data_element('StudyDate').value
            except:
                study_day = None
            try:
                patient_name = str(dsa.data_element('PatientName').value).replace('/', '').replace(' ','_')
            except:
                patient_name =None
            return patient_name, study_day
        except Exception as e:
            return None, None


    def rename(self, root_path):
        for (root, dirs, files) in os.walk(root_path):
            counter = 0
            for dir in dirs:
                if dir in self.ignore_dirs:
                    continue
                counter += 1
                if counter % 100 == 0:
                    print(counter)
                if dir.startswith('p') or dir.startswith('P'):
                    self.rename_dir(root, dir)
            break

    def rename_dir(self, parent_dir, dir_name):
        patient_name = []
        study_day = None
        old_path = os.path.join(parent_dir, dir_name)
        for (root, dirs, files) in os.walk(old_path):
            for file in files:
                name, day = self.read_file(os.path.join(root, file))
                if name is not None and name not in patient_name:
                    patient_name.append(name)
                if study_day is None:
                    study_day = day
        if len(patient_name) > 0:
            patient_names = '-'.join(patient_name)
            if study_day is not None:
                new_dir_name = patient_names + "-" + study_day
            else:
                new_dir_name = patient_names
            # names = dir_name.split('-')
            # names[1] = new_dir_name.replace('/', '')
            new_path = os.path.join(parent_dir, new_dir_name)
            # dir_list = glob.glob(new_path + )
            counter = 0
            rename_new_path = new_path
            if rename_new_path != old_path:
                while os.path.exists(rename_new_path):
                    counter += 1
                    rename_new_path = new_path + "_%d" % counter
                try:
                    os.rename(old_path, rename_new_path)
                except Exception as e:
                    print(old_path, repr(e))



if __name__ == '__main__':
    print(121 % 100)
    import sys
    root_path = sys.argv[1:2]
    rdf = RenameDir(root_path[0])
    rdf.rename(root_path[0])
