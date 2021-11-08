import cv2
import os
import argparse
import random
from tqdm import tqdm
import numpy as np
import multiprocessing
from multiprocessing.managers import BaseManager

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur,GaussianBlur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,RandomBrightness,ToSepia
)


def strong_aug(p=1):
    return Compose([
        HorizontalFlip(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.25),
            GaussianBlur(p=0.5),
            Blur(blur_limit=3, p=0.25),
        ], p=0.2),
        HueSaturationValue(p=0.2),
        OneOf([
            RandomBrightness(),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.6),
        ToSepia(p=0.1)
    ], p=p)

args_parser = argparse.ArgumentParser()


args_parser.add_argument("--input-path", type=str, required=True, help="Directory of train dataset.")

args_parser.add_argument("--prc", type=int, required=False,default=16, help="Directory of train dataset.")





def write_img(img_line,args,pbra):
    aug = strong_aug()
    folder_name = img_line.get()
    if os.path.isdir(os.path.join(args.input_path,folder_name)):
        if not os.path.exists(os.path.join(args.input_path,folder_name+'_enhance')):
            os.makedirs(os.path.join(args.input_path,folder_name+'_enhance'))
            

        # for part in menu:
            # if not os.path.exists(os.path.join(args.input_path,part,folder_name+'_enhance')):
                # os.makedirs(os.path.join(args.input_path,part,folder_name+'_enhance'))
        for file in os.listdir(os.path.join(args.input_path,folder_name)):
            if file.endswith('.png'):
                image = cv2.imread(os.path.join(args.input_path,folder_name,file))
                if image is not None:
                    htich1 = aug(image=image)['image']
                    htich2 = aug(image=image)['image']
                    cv2.imwrite(os.path.join(args.input_path,folder_name+'_enhance','ab_'+file),htich1)
                    cv2.imwrite(os.path.join(args.input_path,folder_name+'_enhance','ab2_'+file),htich2)

                
    pbra.update(1)
    # image = cv2.imread('D:\\Deepfake-GAn\\pngs\\1.png')
# cv2.imshow('res',colorful_img(1,-20,image))
# cv2.waitKey(0)


class MyManager(BaseManager):
    pass
def Manager2():
    m = MyManager()
    m.start()
    return m


MyManager.register('tqdm', tqdm)



if __name__ == '__main__':
    global args
    args = args_parser.parse_args()
    menu = ['face','eyes','nose','mouth']
    img_line = multiprocessing.Manager().Queue()
    manager = Manager2()
    for folder_name in os.listdir(args.input_path):
        if folder_name.endswith('_aligned'):
            img_line.put(folder_name)
    pool = multiprocessing.Pool(args.prc)
    pbra = manager.tqdm(total=img_line.qsize(), ncols=75)
    for i in range(img_line.qsize()):
        res = pool.apply_async(write_img, args=(img_line,args,pbra,))
    pool.close()
    pool.join()
    pbra.close()
    
    print("Done.")
    