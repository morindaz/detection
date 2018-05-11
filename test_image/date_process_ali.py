# -*- coding:utf8- *-
"""
将阿里识别的结果20180230->2018年02月30日
"""
import pandas as pd
import numpy as np
def process_date(date):
        year = date[0:4]
        month = date[4:6]
        day = date[6:8]
        return (year+"年"+month+"月"+day+"日")



if __name__ == '__main__':
    path = "./xlsx/final_ocr_ali.xlsx"
    output_name = "./finale_ali_date.csv"
    file = pd.read_excel(path,header=0)
    establish = file["establish"].values

    # map(lambda x:str(int(x)),establish)
    # print(establish)
    print("load file successfully")
    final_date = []
    for i in establish:
        if pd.isnull(i) or None:
            final_date.append(i)
        else:
            final_date.append(process_date(str(int(i))))
    df = {"date":final_date}
    output = pd.DataFrame(df)
    output.to_csv(output_name)
    print("labels generated to file {} finished".format(output_name))