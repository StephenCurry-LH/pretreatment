import numpy as np
import cv2, os, json, shutil
from copy import deepcopy
from os import path as osp
ncc = 0


#############################################################看这里############################################
'''
原始图像从\\data\\avi_dcm\\source中的文件中解析得到
会保存在  \\data\\avi_img 和 \\data\\dcm_img中
目标图像在\\data\\target中
原始图像向目标图像匹配
'''
# GE_avi:734X1584
#缩放比例：575
#裁剪比例：[56:536,300:948]

# GE_avi:982X1580
#缩放比例：576
#裁剪比例：[56, 536, 141, 789]


#飞利浦_dicom_avi:1130X2196
#缩放比例：476
#裁剪以及padding比例：(resize_label_img[0:476, 140:788, idx], ((2, 2), (0,0))

#飞利浦_dicom_avi:,812X1560
#缩放比例：476
#裁剪以及padding比例：(resize_label_img[0:476, 140:788, idx], ((2, 2), (0,0))

#飞利浦_dicom_avi:,820X1560
#缩放比例：476
#裁剪以及padding比例：(resize_label_img[0:476, 140:788, idx], ((2, 2), (0,0))

#飞利浦_dicom_avi:836X1576
#缩放比例：476
#裁剪以及padding比例：(resize_label_img[0:476, 140:788, idx], ((2, 2), (0,0))


# 放缩和填充函数
def pad_scale_label(src):
    h, w, c = src.shape
    scale_h = 0  # 放缩比例
    if h == 982 :#GE的
        scale_h = 576
        index = [56, 536, 141, 789]
        pad_index = [0, 0, 0, 0]
        # print(src.shape)
    elif h == 734:  # GE的
        scale_h = 576
        index = [56,536,298,946]
        pad_index=[0,0,0,0]
        # print(src.shape)
    elif h == 820:  # 飞利浦的
        scale_h = 476
        index = [0, 476, 142, 790]
        pad_index = [2, 2, 0, 0]
        # print(src.shape)
    elif h ==1130  :  # 飞利浦的
        scale_h = 476
        index = [0, 476, 152, 800]
        pad_index = [1, 3, 0, 0]
        # print(src.shape)
    elif h == 812:  # 飞利浦的
        scale_h = 476
        index = [0, 476, 145, 793]
        pad_index = [2, 2, 0, 0]

    elif h == 836:  # 飞利浦的
        scale_h = 476
        index = [0, 476, 138, 786]
        pad_index = [2, 2, 0, 0]
     # print(src.shape)

    else:
        scale_h = 0

    if scale_h == 0:
        print("需要自己重新计算放缩比例")
    h, w = cal_scale(h, w, scale_h)  # 调用放缩函数
    resize_label_img = cv2.resize(src, (w, h))
    pad_label_img = np.zeros((480, 648, 3))
    for idx in range(resize_label_img.shape[2]):
        pad_label_ch = np.pad(resize_label_img[index[0]:index[1],index[2]:index[3], idx], ((pad_index[0],pad_index[1]), (pad_index[2],pad_index[3])), 'constant', constant_values=(0, 0))
        pad_label_img[:, :, idx] = pad_label_ch
    return pad_label_img



def cal_scale(h, w, new_h):
    return new_h, int(w * new_h / h)


def save_scaled_label(path_to_label, save_dir):
    labels = sorted(os.listdir(path_to_label))
    for label in labels:
        label_dir = osp.join(path_to_label, label)
        label_img = cv2.imread(label_dir)
        pad_label_img = pad_scale_label(label_img)
        cv2.imwrite(os.path.join(save_dir, label),pad_label_img)
#############################################################################################################


if __name__ == '__main__':
    src_dir = 'data\\avi_img'
    target_shape = [648, 480]
    for device in os.listdir(src_dir):
        device_dir = os.path.join(src_dir, device)
        for root, dirs, files in os.walk(device_dir):
            for patient in dirs:
                if 'MLX' in patient or '08-' in patient:#判断是病人
                    patient_dir = os.path.join(root, patient)
                    for data_dir in os.listdir(patient_dir):
                        path_to_label = os.path.join(patient_dir, data_dir)
                        path_to_avi = path_to_label
                        path_to_save_label = path_to_label.replace('avi_img', 'label')
                        if not os.path.exists(path_to_save_label):
                            os.makedirs(path_to_save_label)
                        save_scaled_label(path_to_avi, path_to_save_label)



