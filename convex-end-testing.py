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
#testing


class Nstr:
    def __init__(self, arg):
       self.x=arg
    def __sub__(self,other):
        c=self.x.replace(other.x,"")
        return c

def max_contour(contours):#用于找到图片中面积最大的图形并返回
    areas = []
    if len(contours) == 0:
        return -1;
    else :

        for c in range(len(contours)):
            areas.append(cv2.contourArea(contours[c]))
        max_id = areas.index(max(areas))
        #max_rect = cv2.minAreaRect(contours[max_id])
        return max_id


def read(read_path,read_name,num):
    data = xd.open_workbook(read_path)
    sheet = data.sheet_by_name(read_name)
    data_all = []
    read_list = []
    for r in range(sheet.nrows):
        data_list = []
        for c in range(sheet.ncols):
            data_list.append(sheet.cell_value(r,c))
        data_all.append(data_list)
    for list in data_all:
        read_list.append(int(list[num]))
    return read_list

# def read_ecg(file):
#     with open(file, 'r') as f:
#         array = []
#         for line in f.readlines()[0:]:  # 从第二行开始计数
#             line = [int(x) for x in line.split()]
#             array.extend(line)
#         array = np.array(array).reshape(2048, -1)  # 5000*8
#     return array

#read_list_one = read("/home/lihang/pretreatment/CEW舒张功能  名单.xlsx","CEW舒张功能  名单",20)
#k= len(read_list_one)
img_save = '/home/lihang/pretreatment/line/'
xlsx_save = '/home/lihang/pretreatment/xlsx/'
npy_save = '/home/lihang/pretreatment/npy_data/'
txt_save = '/home/lihang/pretreatment/txt_data/'
img_save_plt = '/home/lihang/pretreatment/plt/rectArea'
img_root = '/home/lihang/pretreatment/out/'

dir_name = os.listdir(img_root)
for name in dir_name:
    #print(name)

    img_folder = os.path.join(img_root,name)
   # print(img_folder)
    img_list = [os.path.join(img_folder, nm) for nm in os.listdir(img_folder) if nm[-3:] in ['jpg', 'png', 'gif']]
    img_list.sort()
    # for imging in img_list:
    #     print(imging)
    list_all = []
    j = 0
    data_list = []
    x_list = []
    y_list = []
    for imgs in img_list:
        x_list.append(int(j))
        # if (j >= k):
        #     print("error")
        #     break
        # else:
        m = Nstr(img_folder)
        n = Nstr(imgs)
        sub = n - m
        list = []
        img = cv2.imread(imgs, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 二值化,转成灰度图
        ret, thresh = cv2.threshold(gray, 127, 255,
                                    cv2.THRESH_BINARY)  # 图片轮廓，ret, dst = cv2.thresh(src, thresh, maxval, type)， src表示输入的图片， thresh表示阈值， maxval表示最大值， type表示阈值的类型
        # cv2.THRESH_BINARY   表示阈值的二值化操作，大于阈值使用maxval表示，小于阈值使用0表示
        contours, hierarchy = cv2.findContours(thresh, 2, 1)
        # contours, hierarchy = cv.findContours( image, mode, method[, contours[, hierarchy[, offset]]] )，参数1：源图像，参数2：轮廓的检索方式，参数3：一般用 cv.CHAIN_APPROX_SIMPLE，就表示用尽可能少的像素点表示轮廓
        # contours：图像轮廓坐标，是一个链表
        #cnt = contours[0]
        max_cnt = max_contour(contours)
        if max_cnt == -1:
            continue
        else :

            cnt = contours[max_cnt]
           # cnt = max_contour(contours)
            # 寻找凸包并绘制凸包（轮廓）
            cntArea = cv2.contourArea(cnt)  # 对象图像轮廓面积
            hull = cv2.convexHull(cnt)
            hullArea = cv2.contourArea(hull)  # 凸包面积
            #solidity = float(cntArea) / hullArea  # 对象面积/凸包面积
            length = len(hull)  # 凸包直线个数
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)  # 矩形框
            aspectRatio = float(w) / h  # 矩形框的宽度/高度
            high = float(h)
            weight = float(w)
            rectArea = w * h  # 矩形框的面积
            #extend = float(cntArea) / rectArea  # 对象面积/矩形边界面积
            x = np.array(list)
            y_list.append(high)
            #list.append(sub)
            list.append(hullArea)
            list.append(rectArea)
            list.append(high)
            list.append(weight)

            #list.append(read_list_one[j])

            list_all.append(list)
            j = j + 1
            for i in range(len(hull)):
                cv2.line(img, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), (0, 255, 0), 1)

            save_name = os.path.join(img_save,name)
            #print(save_name)
            if not os.path.exists(save_name):
                os.makedirs(save_name)
            save_name_end = save_name +sub
            #print(save_name_end)
            cv2.imwrite(save_name_end, img)

            # cv2.imshow('line', img)
            # cv2.waitKey(100)

    name_npy = name + '.npy'
    save_name_npy = os.path.join(npy_save,'sample/'+name_npy)

    name_resample_npy = name + '_resample.npy'
    save_name_resample_npy = os.path.join(npy_save,'resample/'+name_resample_npy)
    #print(type(list_all))
    np.save(save_name_npy, list_all)
    print(save_name_npy + " had been written!")
    plt_name = name + 'plt.jpg'
    save_plt_name = os.path.join(img_save_plt,plt_name)


    boxes = np.load(save_name_npy)
    #print(boxes)

    name_txt = name + '.txt'
    save_name_txt = os.path.join(txt_save,'sample/'+name_txt)
    np.savetxt(save_name_txt, boxes, fmt='%s', newline='\n')
    print(save_name_txt + " had been written!")

    name_resample_txt = name+'_resample.txt'
    save_name_resample_txt = os.path.join(txt_save, 'resample/'+name_resample_txt)



    array=np.load(save_name_npy)

    for num in range(0, 3):
        #ecg=array[:,0]
        #ecg_resampled = signal.resample(ecg, 2048)
        ecg_resampled=signal.resample(array,2048)
        x_ecg_resampled=[i for i in range(ecg_resampled.shape[0])]
        np.save(save_name_resample_npy, ecg_resampled)
        print(save_name_resample_npy + " had been written!")
        boxes = np.load(save_name_resample_npy)
        #print(boxes)

        np.savetxt(save_name_resample_txt, boxes, fmt='%s', newline='\n')
        print(save_name_resample_txt + " had been written!")

        array = np.load(save_name_npy)
        ecg = array[:, 1]
        x_ecg = [i for i in range(ecg.shape[0])]
        array_resample = np.load(save_name_resample_npy)
        ecg_resampled = array_resample[:,1]
        x_ecg_resampled = [i for i in range(ecg_resampled.shape[0])]
        plt.subplot(2, 1, 1)
        plt.plot(x_ecg, ecg)
        plt.subplot(2, 1, 2)
        plt.plot(x_ecg_resampled, ecg_resampled)
        #plt.imshow()

        plt.savefig(save_plt_name)
        print(save_plt_name + " had been drawn!")
        plt.show(block=False)

print('sucessful')