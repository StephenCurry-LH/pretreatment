
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
img_save_plt = '/home/lihang/pretreatment/show-curve-single/'
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
        list.append(high*300)
        list.append(weight*300)
        list_all.append(list)
        j = j + 1
        for i in range(len(hull)):
            cv2.line(img, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), (0, 255, 0), 1)

        save_name = os.path.join(img_save, name)
        if not os.path.exists(save_name):
            os.makedirs(save_name)
        save_name_end = save_name + sub
        cv2.imwrite(save_name_end, img)

    plt_name = name + '.jpg'
    save_plt_name = os.path.join(img_save_plt, plt_name)


    array = np.array(list_all)
    x_ecg = [i for i in range(array.shape[0])]
    #plt.subplot(2, 1, 1)
    plt.plot(x_ecg, array[:,0])
    plt.plot(x_ecg, array[:, 1])
    # plt.subplot(2, 1, 2)
    plt.plot(x_ecg, array[:, 2])
    plt.plot(x_ecg, array[:, 3])
    plt.savefig(save_plt_name)
    plt.close()  # 有效解决了画图时多条曲线（也就是上一张图的曲线残留）重叠问题
    print(save_plt_name + " had been drawn!")
    plt.show(block=False)

    #np.save(save_name_npy, list_all)




    # plt.savefig(save_plt_name)
    # plt.close()  # 有效解决了画图时多条曲线（也就是上一张图的曲线残留）重叠问题
    # print(save_plt_name + " had been drawn!")
    # plt.show(block=False)


print('sucessful')


#list  ->  np.load(array)   ->   截断 ->   save