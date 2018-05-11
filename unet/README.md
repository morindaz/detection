unet用于图像分割
训练集：DSA图片   标签：像素级别的标注图片
1、data_process中包含了处理数据集的内容。
需要修改的地方为：
class dataProcess(object):
    def __init__(self, out_rows, out_cols, data_path="/home/maoshunyi/moyamoya/train_img/ap_train_1", label_path="./data/train/label",
                 test_path="/home/maoshunyi/moyamoya/train_img/ap_train_1", npy_path="./npydata", img_type="jpeg"):


2、unet.py中包含了unet的网络结构定义。训练模型并且将测试的结果存储到对应文件夹内。
3、test_predict_me 中通过加载训练好的hdf5文件对于图片进行预测，得到输出图片