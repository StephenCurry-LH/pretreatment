import os
import SimpleITK as sitk
import cv2
import pydicom


def dicom2array(dir):
    reader = sitk.ImageFileReader()
    reader.SetFileName(dir)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)
    return image_array


def dicom2jpg(file_dir, save_dir):
    img_array = dicom2array(file_dir)
    for i in range(img_array.shape[0]):
        img = img_array[i, :, :, :]
        img_name = '{:0>3d}.jpg'.format(i)
        cv2.imwrite(os.path.join(save_dir, img_name), img)
class Nstr:
    def __init__(self, arg):
       self.x=arg
    def __sub__(self,other):
        c=self.x.replace(other.x,"")
        return c
if __name__ == '__main__':
    src_dir = '/home/lihang/pretreatment/CABG/'
    save_dir = "/home/lihang/pretreatment/data/"
    name_dir = "/home/lihang/pretreatment/CABG/"
    name_after = "A4C/DICOM/00001/"
    for root, dirs, files in os.walk(src_dir, topdown=False):
        for name in files:
            file_dir = os.path.join(root, name)
            #print(file_dir)
            # print(type(name))
            # print(type(name_dir))
            m = Nstr(file_dir)
            n = Nstr(name)
            p = Nstr(name_dir)
            k = Nstr(name_after)
            sub = m - n
            o = Nstr(sub)
            #print(type(sub))
            sub1 = o- p
            i = Nstr(sub1)
            sub_all = i - k
            #sub = sub - name_dir

            # print(name)
            # print(sub)
            # print(sub_all)

            save_end = save_dir + sub_all
            #save_end1 = save_end + '\'
            # print(save_end)
            # print(save_dir)
            # print(type(save_end))
            # print(type(save_dir))
            if not os.path.exists(save_end):
                os.makedirs(save_end)
            dicom2jpg(file_dir, save_end)
print("sucessfully")