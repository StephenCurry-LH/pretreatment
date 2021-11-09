# import numpy as np
# from scipy.signal import argrelextrema
# import pylab as pl
# import matplotlib.pyplot as plt
# import scipy.signal as signal
# from scipy import signal
# import os
# # def maximum(array):
# #     length = len(array)
# #     max = []
# #     for i in range(0,length):
# #         if
# #         if()
# #
# #     for
# def two_max(array):
#     x=x
#     peaks, _ = signal.find_peaks(array, distance=30)
#     return array[peaks[0],peaks[2]]
#
# if __name__ == '__main__':
#     # x = np.array([
#     #     0, 6, 25, 20, 15, 8, 15, 6, 0, 6, 0, -5, -15, -3, 4, 10, 8, 13, 8, 10, 3,
#     #     1, 20, 7, 3, 0])
#     # print(x[2:5])
#     # print(type(x))
#     # # y = signal.resample(x,(0:5))
#     # print(x[2:6])
#     root_name = "C://Users//HP//Desktop//pretreatment//picture//"
#     files = os.listdir(root_name)
#     for file in files:
#         #print(file)
#         p = []
#         save_name_npy  = os.path.join(root_name,file)
#         #print(save_name_npy)
#         array = np.load(save_name_npy)
#         print(type(array))
#         #ecg = array[:,2]
#         #print(ecg)
#     #     #plt.figure(figsize=(50,10))
#     #     # print(peaks)
#     #     peaks, _ = signal.find_peaks(ecg, distance=30)
#     # #     p.append(len(peaks))
#     # # print(len(p))
#     #     print(peaks)
#     #     print(len(peaks))
#     #     plt.plot(np.arange(len(ecg)),ecg)
#     #     plt.show()
#
#
#     #print(maximum(x))
#     #print(argrelextrema(x,np.greater())  )
#
# # plt.figure(figsize=(16,4))
# # plt.plot(np.arange(len(x)),x)
# # print(x[signal.argrelextrema(x, np.greater)])
# # print(signal.argrelextrema(x, np.greater))
#
# # plt.plot(signal.argrelextrema(x,np.greater)[0],x[signal.argrelextrema(x, np.greater)],'o')
# # plt.plot(signal.argrelextrema(-x,np.greater)[0],x[signal.argrelextrema(-x, np.greater)],'+')
# # # plt.plot(peakutils.index(-x),x[peakutils.index(-x)],'*')
# # plt.show()
#
#
# # peaks, _ = signal.find_peaks(y, distance=5) #distance表示极大值点两两之间的距离至少大于等于5个水平单位
# #
# # print(peaks)
# #
# # print(len(peaks))  # the number of peaks
# # '''
# # 79
# # '''
# #
# # plt.figure(figsize=(20,5))
# # plt.plot(y)
# # for i in range(len(peaks)):
# #     plt.plot(peaks[i], y[peaks[i]],'*',markersize=10)
# # plt.show()
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
save_name_npy = '/home/lihang/pretreatment/npy_data_end/0_CEW-11.npy'
save_name_txt = '/home/lihang/pretreatment/test1.txt'
boxes = np.load(save_name_npy)

np.savetxt(save_name_txt, boxes, fmt='%s', newline='\n')