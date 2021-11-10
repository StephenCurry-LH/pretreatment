
import csv
import xlsxwriter
import cv2
import xlwt
from numpy import array
from openpyxl import Workbook
import numpy as np
import numpy as np
import xlrd as xd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# testing


class Nstr:
    def __init__(self, arg):
        self.x = arg

    def __sub__(self, other):
        c = self.x.replace(other.x, "")
        return c

def maxminnorm(array):
    maxcols = array.max(axis = 0)
    mincols = array.min(axis = 0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t = np.empty((data_rows,data_cols))
    for i in range(data_cols):
        t[:,i] = (array[:,i] - mincols[i]) / (maxcols[i] - mincols[i])
    return t

def max_contour(contours):  # 用于找到图片中面积最大的图形并返回
    areas = []
    for c in range(len(contours)):
        areas.append(cv2.contourArea(contours[c]))
    max_id = areas.index(max(areas))
    # max_rect = cv2.minAreaRect(contours[max_id])
    return contours[max_id]


def read(read_path, read_name, num):
    data = xd.open_workbook(read_path)
    sheet = data.sheet_by_name(read_name)
    data_all = []
    read_list = []
    for r in range(sheet.nrows):
        data_list = []
        for c in range(sheet.ncols):
            data_list.append(sheet.cell_value(r, c))
        data_all.append(data_list)
    for list in data_all:
        read_list.append(int(list[num]))
    return read_list

img_save = '/home/lihang/pretreatment/line-ending/'
npy_save = '/home/lihang/pretreatment/npy-ending/'
txt_save = '/home/lihang/pretreatment/txt-ending/'
img_save_plt = '/home/lihang/pretreatment/plt-ending/'
img_root = '/home/lihang/pretreatment/data_end/'

dir_name = os.listdir(img_root)
for name in dir_name:
    img_folder = os.path.join(img_root, name)
    img_list = [os.path.join(img_folder, nm) for nm in os.listdir(img_folder) if nm[-3:] in ['jpg', 'png', 'gif']]
    img_list.sort()
    list_all = []
    j = 0
    data_list = []
    x_list = []
    y_list = []
    for imgs in img_list:
        x_list.append(int(j))
        m = Nstr(img_folder)
        n = Nstr(imgs)
        sub = n - m
        list = []
        img = cv2.imread(imgs, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255,
                                    cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, 2, 1)
        cnt = max_contour(contours)
        cntArea = cv2.contourArea(cnt)  # 对象图像轮廓面积
        hull = cv2.convexHull(cnt)
        hullArea = cv2.contourArea(hull)  # 凸包面积
        length = len(hull)  # 凸包直线个数
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)  # 矩形框
        aspectRatio = float(w) / h  # 矩形框的宽度/高度
        high = float(h)
        weight = float(w)
        rectArea = w * h  # 矩形框的面积
        x = np.array(list)
        y_list.append(high)
        # list.append(sub)
        list.append(hullArea)
        list.append(rectArea)
        list.append(high)
        list.append(weight)
        list_all.append(list)
        j = j + 1
        for i in range(len(hull)):
            cv2.line(img, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), (0, 255, 0), 1)

        save_name = os.path.join(img_save, name)
        if not os.path.exists(save_name):
            os.makedirs(save_name)
        save_name_end = save_name + sub
        cv2.imwrite(save_name_end, img)

    name_npy = name + '.npy'
    save_name_npy = os.path.join(npy_save, name_npy)

    #np.save(save_name_npy, list_all)
    plt_name = name + 'plt.jpg'
    save_plt_name = os.path.join(img_save_plt, plt_name)

    name_txt = name + '.txt'
    save_name_txt = os.path.join(txt_save, name_txt)

    array = np.array(list_all)
    ecg_hull = array[:,0]
    ecg_react = array[:,1]
    ecg_height = array[:,2]
    ecg_weight = array[:,3]

    #ecg = array[:, 1]
    peaks, _ = signal.find_peaks(ecg_react, distance=25)
    ecging_hull = ecg_hull[peaks[1]:peaks[2]]
    ecging_react = ecg_react[peaks[1]:peaks[2]]
    ecging_height = ecg_height[peaks[1]:peaks[2]]
    ecging_weight = ecg_weight[peaks[1]:peaks[2]]
    #ecging = ecg[peaks[0]:peaks[2]]

    ecg_resample_hull = signal.resample(ecging_hull, 512)
    ecg_resample_react = signal.resample(ecging_react, 512)
    ecg_resample_height = signal.resample(ecging_height, 512)
    ecg_resample_weight = signal.resample(ecging_weight, 512)

    ecg_all = []
    ecg_all.append(ecg_resample_hull)
    ecg_all.append(ecg_resample_react)
    ecg_all.append(ecg_resample_height)
    ecg_all.append(ecg_resample_weight)

    # ecg_all.append(ecging_hull)
    # ecg_all.append(ecging_react)
    # ecg_all.append(ecging_height)
    # ecg_all.append(ecging_weight)

    ecg_all_trans = np.transpose(ecg_all)
    ecg_all_trans_normal = maxminnorm(ecg_all_trans)
    np.save(save_name_npy,ecg_all_trans_normal)

    print(save_name_npy + " had been writtern")
    boxes = np.load(save_name_npy)

    np.savetxt(save_name_txt, boxes, fmt='%s', newline='\n')
    print(save_name_txt + " had been writtern")

    x_ecg = [i for i in range(ecg_all[0].shape[0])]
    # plt.subplot(2, 1, 1)
    # plt.plot(x_ecg, ecg_all[0])
    # plt.plot(x_ecg, ecg_all[1])

    x_ecg_origin = [i for i in range(array.shape[0])]
    # plt.subplot(2, 1, 1)
    # plt.plot(x_ecg_origin, array[:, 0])
    # plt.plot(x_ecg_origin, array[:, 1])
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(x_ecg, ecg_all[0])
    # plt.plot(x_ecg, ecg_all[1])

    #plt.subplot(2, 1, 1)
    plt.plot(x_ecg, ecg_all_trans_normal[0])
    plt.plot(x_ecg, ecg_all_trans_normal[1])
    #plt.subplot(2, 1, 2)
    plt.plot(x_ecg, ecg_all_trans_normal[2])
    plt.plot(x_ecg, ecg_all_trans_normal[3])

    plt.savefig(save_plt_name)
    plt.close()  # 有效解决了画图时多条曲线（也就是上一张图的曲线残留）重叠问题
    print(save_plt_name + " had been drawn!")
    plt.show(block=False)


print('sucessful')


#list  ->  np.load(array)   ->   截断 ->   save