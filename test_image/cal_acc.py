# -*- coding: utf-8 -*-
"""
用于计算样本和高研院以及阿里识别的准确率情况
"""

import pandas as pd
data_file_ali = "./xlsx/ali.xlsx"
data_file_gaoyan = "./xlsx/gaoyan.xlsx"
data_file_label = "./xlsx/final_ocr.xlsx"
txt_file = "./xlsx/top_30.txt"

def read_file(file):
    data_file =pd.read_excel(file, header=0)
    data_address = data_file["address"].values[txt]
    data_company = data_file["company_name"].values[txt]
    data_establish = data_file["establish"].values[txt]
    data_person = data_file["person"].values[txt]
    data_reg_num = data_file["reg_num"].values[txt]
    return data_address,data_company,data_establish,data_person,data_reg_num,data_file

def read_txt(txt):
    total_txt = []
    with open(txt, 'r') as f:
        data = f.readlines()  # txt中所有字符串读入data
        print(data)
    for i in data:
        total_txt.append(int(i.strip()))
    print(total_txt)
    return total_txt


def cal_each_acc(data_ali,data_gaoyan,label,name):
    each_count_ali = 0
    each_count_gaoyan = 0
    whole_count = len(data_ali)
    for i in range(whole_count):
        if data_ali[i] == label[i]:
            each_count_ali += 1
        if data_gaoyan[i] == label[i]:
            each_count_gaoyan += 1
    acc_ali = round(each_count_ali/whole_count*100,2)
    acc_gaoyan = round(each_count_gaoyan/whole_count*100,2)
    print("ali_{}:{}%".format(name,acc_ali))
    print("gaoyan_{}:{}%".format(name,acc_gaoyan))
    print("------------------")
    return acc_ali,acc_gaoyan


txt = read_txt(txt_file)

data_ali_address,data_ali_company,data_ali_establish,\
data_ali_person,data_ali_reg_num,ali_file = read_file(data_file_ali)

data_gaoyan_address,data_gaoyan_company,data_gaoyan_establish,\
data_gaoyan_person,data_gaoyan_reg_num,gaoyan_file = read_file(data_file_gaoyan)

data_label_address, data_label_company,data_label_establish,\
data_label_person, data_label_reg_num,label_file = read_file(data_file_label)

address_acc_ali, address_acc_gaoyan = cal_each_acc(data_ali_address,data_gaoyan_address,data_label_address,"address_acc")
company_acc_ali, company_acc_gaoyan = cal_each_acc(data_ali_company,data_gaoyan_company,data_label_company,"company_acc")
establish_acc_ali, establish_acc_gaoyan = cal_each_acc(data_ali_establish,data_gaoyan_establish,data_label_establish,"establish_acc")
person_acc_ali, person_acc_gaoyan = cal_each_acc(data_ali_person,data_gaoyan_person,data_label_person,"person_acc")
reg_num_acc_ali, reg_num_acc_gaoyan = cal_each_acc(data_ali_reg_num,data_gaoyan_reg_num,data_label_reg_num,"reg_num_acc")


data_ali =ali_file.values[txt,:]
data_gaoyan = gaoyan_file.values[txt,:]
data_label = label_file.values[txt,:]
print("load file successfully")
ali_count = 0
gaoyan_count = 0
global_count = 0
for i in range(len(data_ali)):
    # gao = "".join(map(lambda x:str(x),data_gaoyan[i]))
    # ali = "".join(map(lambda x:str(x),data_ali[i]))
    # truth = "".join(map(lambda x:str(x),data_label[i]))
    for j in range(len(data_ali[i])):
        global_count +=1
        ali  =data_ali[i][j]
        gaoyan = data_gaoyan[i][j]
        label = data_label[i][j]
        # print("ali",ali)
        # print("gaoyan",gaoyan)
        # print("label",label)
        if  ali== label:
            ali_count +=1
        if gaoyan == label:
            gaoyan_count += 1
        else:
            pass
            # print("gaoyan",gaoyan)
            # print("label",label)


print("ali_whole_acc:{}%".format(ali_count/global_count*100))
print("gaoyan_whole_acc:{}%".format(round(gaoyan_count/global_count*100,2)))
print("global_cnt",global_count)



