from PIL import Image
import os
dir_path = '/home/lihang/pretreatment/outing/'
for root ,dirs,files in os.walk(dir_path):
    for file in files:
        file_name = os.path.join(root,file)
        #print(file_name)
        im = Image.open(file_name)
        print(file_name)
        print(im.size[0],im.size[1])

print("sucessfully")
# im = Image.open(filename)#返回一个Image对象
# print('宽：%d,高：%d'%(im.size[0],im.size[1]))