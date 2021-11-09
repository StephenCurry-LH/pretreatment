import numpy as np
import cv2
import os

class Nstr:
    def __init__(self, arg):
       self.x=arg
    def __sub__(self,other):
        c=self.x.replace(other.x,"")
        return c

def update(input_img_path, output_img_path):#图片裁剪
    image = cv2.imread(input_img_path)
    #a = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REFLECT)
    #print(image.shape)
    cropped = image[60:540, 76:724]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite(output_img_path, cropped)

def expand(input_img_path,output_img_path):
    image = cv2.imread(input_img_path)
    a = cv2.copyMakeBorder(image, 23, 23, 6, 6, cv2.BORDER_REPLICATE)
    #根据图像的边界的像素值，向外扩充图片，左右扩充7个像素  上下扩充14个像素
    cv2.imwrite(output_img_path,a)


output_dir = '/home/lihang/pretreatment/outing-test/'
dir_path = '/home/lihang/pretreatment/data-in/'
for root,dirs,files in os.walk(dir_path):

    if(root == '/home/lihang/pretreatment/data-in/'):
        continue
    else:
        m = Nstr(root)
        n = Nstr(dir_path)
        p = m - n
        out_dir = output_dir + p
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

# 获得需要转化的图片路径并生成目标路径
        image_filenames = [(os.path.join(root, x), os.path.join(out_dir, x))
                       for x in os.listdir(root)]
    # # 转化所有图片
        for path in image_filenames:
            expand(path[0], path[1])
            print(path[0] + "had been resize!")
    print("The direction of " + root + 'had been resize!!!!!!!!!')
print("sucessfully")


