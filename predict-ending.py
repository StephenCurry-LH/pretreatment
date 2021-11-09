import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
#111111111
from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
class Nstr:
    def __init__(self, arg):
       self.x=arg
    def __sub__(self,other):
        c=self.x.replace(other.x,"")
        return c

def predict_img(net,
                full_img,
                device,
                # scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                L.append(os.path.join(root, file))
    return L
if __name__ == "__main__":
    in_files = "/home/lihang/pretreatment/outing/"
    out_files = "/home/lihang/pretreatment/out-resize/"


    net = UNet(n_channels=3, n_classes=1)

    device = torch.device('cpu')

    net.to(device=device)
    net.load_state_dict(torch.load('checkpoints/LV_PHILIPS_A4C/CP_epoch10.pth', map_location=device))

    files_name = file_name(in_files)
    for file_name in files_name:
        #print(file_name)
        m = Nstr(file_name)
        n = Nstr(in_files)
        sub = m - n
        #print(sub)
        (filepath, tempfilename) = os.path.split(file_name)

        p = Nstr(sub)
        o = Nstr(tempfilename)
        target_path = p - o
        #print(target_path)
        img = Image.open(file_name)

        mask = predict_img(net=net,
                           full_img=img,
                           # scale_factor=args.scale,
                           out_threshold= 0.5,
                           device=device)

        target_path_end = os.path.join(out_files,target_path)
        #print(target_path_end)

        if not os.path.exists(target_path_end):
            os.mkdir(target_path_end)

        result = mask_to_image(mask)
        result_path=os.path.join(out_files,sub)
        result.save(result_path)
        print(file_name + "  had been cut!")
    #print("the picture of" + target_path_end + "had been cut sucessfully!")


print("cut sucessfully")