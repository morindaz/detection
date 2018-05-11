# -*- coding: utf-8 -*-
import glob
import json
import os
import pandas as pd
path = "./qiu/label_gaoyan"
json_files = glob.glob(path+"/*")
output_name = "ocr_label_gaoyan.csv"
# with open(template_json, 'r', encoding='utf-8-sig') as h:
#     template = json.load(h)
img_name = [] #图像名字
company_name = [] #公司名称
business = [] #经营范围
establish = [] #成立日期
reg_num = []  #统一社会信用代码
address = []  #营业场所
person = [] #负责人
count_number = []
count =0
for json_file in json_files:
    print(count)

    _, json_name = os.path.split(json_file)
    img_name.append(json_name)
    print(json_name)
    with open(json_file, 'r', encoding='utf-8-sig') as single_json:
        template = json.load(single_json)
        if "名称" in template:
            company_name.append(template["名称"])
        else:
            company_name.append("")
        if "经营范围" in template:
            business.append(template["经营范围"])
        else:
            business.append("")
        if "成立日期" in template:
            establish.append(template["成立日期"])
            print(template["成立日期"])
        else:
            establish.append("")
        if "统一社会信用代码" in template:
            reg_num.append(template["统一社会信用代码"])
        else:
            reg_num.append("")
        if "住所" in template:
            address.append(template["住所"])
        else:
            address.append("")
        if "法定代表人" in template:
            person.append(template["法定代表人"])
        else:
            person.append("")
        flag =  json_name.split('.')[0]
        count_number.append(flag)
        count += 1

print(json_files)
#
# df = {"json_name": img_name, "company_name":company_name, "establish": establish,
#       "reg_num":reg_num,"address":address,"person":person,"count":count_number}
# output = pd.DataFrame(df)
# output.to_csv(output_name)
# print("labels generated to file {} finished".format(output_name))


