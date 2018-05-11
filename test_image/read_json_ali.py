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
        if "name" in template:
            company_name.append(template["name"])
        else:
            company_name.append("")
        if "business" in template:
            business.append(template["business"])
        else:
            business.append("")
        if "establish_date" in template:
            establish.append(template["establish_date"])
        else:
            establish.append("")
        if "reg_num" in template:
            reg_num.append(template["reg_num"])
        else:
            reg_num.append("")
        if "address" in template:
            address.append(template["address"])
        else:
            address.append("")
        if "person" in template:
            person.append(template["person"])
        else:
            person.append("")
        flag =  json_name.split('.')[0]
        count_number.append(flag)
        count += 1

print(json_files)

df = {"json_name": img_name, "company_name":company_name, "establish": establish,
      "reg_num":reg_num,"address":address,"person":person,"count":count_number}
output = pd.DataFrame(df)
output.to_csv(output_name)
print("labels generated to file {} finished".format(output_name))


