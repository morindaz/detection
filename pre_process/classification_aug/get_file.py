"""
There's a failure in the data_aug_class file when trying to save the
file_name,type and path.
The file name is not the same length as the others.
But when I run locally, it's fine. However, there's some bugs on the server
"""
import glob
import os
import pandas as pd
img_input_dir = "./image_gen/"
output_name = "./aug_images.csv"
img_origin = glob.glob(img_input_dir + "*") #get the file names
whole_path = []
whole_name = []
whole_type = []
for i in img_origin:
    _, type_name = os.path.split(i)
    per_dir = img_input_dir+type_name
    image_in_dir = glob.glob(per_dir+"/*")
    dir_part_path =[] # image path in every file
    dir_part_name =[] #image name in every file
    dir_part_type = []
    for j in image_in_dir:
        img_save_dir,img_save_name = os.path.split(j)
        dir_part_path.append(img_save_dir)
        dir_part_name.append(img_save_name)
        dir_part_type.append(type_name)

    whole_path.extend(dir_part_path)
    whole_name.extend(dir_part_name)
    whole_type.extend(dir_part_type)

df = {"filename": whole_name, "type": whole_type, "address": whole_path}
output = pd.DataFrame(df)
output.to_csv(output_name)
print("labels generated to file {} finished".format(output_name))


