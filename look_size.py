import os
from os import path as osp
import cv2

data_path = "data\\dcm_img"

for device_type in os.listdir(data_path):
    if 'PHILIPS' in device_type:
        print('PHILIPS')
        patients = osp.join(data_path, device_type)
        for patient in os.listdir(patients):
            img_root = osp.join(patients, patient)
            for root, dirs, files in os.walk(img_root, topdown=False):
                for ind, file in enumerate(files):
                    if ind == 0:
                        # print(osp.join(root, file))
                        img = cv2.imread(osp.join(root, file))
                        # if img.shape[0] == 1130:
                        print('\t',patient, img.shape)
    if 'GE' in device_type:
        print('GE')
        patients = osp.join(data_path, device_type)
        for patient in os.listdir(patients):
            img_root = osp.join(patients, patient)
            for root, dirs, files in os.walk(img_root, topdown=False):
                for ind, file in enumerate(files):
                    if ind == 0:
                        # print(osp.join(root, file))
                        img = cv2.imread(osp.join(root, file))
                        if img.shape[0] == 708:
                             print('\t', patient, img.shape)
