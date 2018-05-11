# -*- coding: utf-8 -*-
"""
This file is used to deal with the date
mainly for gaoyan

"""
import pandas as pd
import numpy as np

# XX年X月X日->20180324
def process_date(date):
    final_date = ""
    if "年" in date:
        a1 =date.split("年")
        year = a1[0]
        rest = a1[1]
        final_date +=year+"年"
    else:
        return date
    if "月" in rest:
        a2 = rest.split("月")
        month = a2[0]
        rest = a2[1]
        if pd.isnull(month) or month=='':
            return date
        elif int(month) < 10:
            month = "0" + month
        final_date += month+"月"
    else:
        return date
    if "日" in date:
        a3 = rest.split("日")
        day = a3[0]
        if int(day)<10:
            day = "0"+day
        final_date += day+"日"
    else:
        return date
    print(final_date)
    return (final_date)

if __name__ == '__main__':
    path = "./xlsx/gaoyan.xlsx"
    output_name = "./gaoyan_date.csv"
    # data_file_gaoyan = "./gaoyan.xlsx"
    # data_file_label = "./final_ocr_ali.xlsx"
    data_gaoyan = pd.read_excel(path, header=0)
    dates = data_gaoyan["establish"].values
    final_date = []
    count =0
    for i in dates:
        # print(count,"------------")
        if pd.isnull(i) or None or '':
            final_date.append(i)
            # print("is null",i)
        else:
            ret = process_date(i)
            final_date.append(ret)
        count +=1
    # print(final_date)
    df = {"date":final_date}
    output = pd.DataFrame(df)
    output.to_csv(output_name)
    print("labels generated to file {} finished".format(output_name))
