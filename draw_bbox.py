#!/usr/bin/env python
# coding: utf-8
"""
Created on April, 2019
@authors: Hulking
"""
import os
import cv2
import glob
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input as preprocess
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pylab
from PIL import Image, ImageFont, ImageDraw

"""
路径定义
"""
#path list
anchors_path='./model_data/yolo_anchors.txt'
classes_path='./model_data/coco_classes.txt'
img_list_path='./model_data/5k.txt'
img_list_dir="/home/common/datasets/coco/"
imgs_path='/home/common/datasets/coco/images/val2014/'
gt_folder= '/home/common/datasets/coco/annotations/'
res_path='./results/cocoapi_results.json'
res_dir='./results/'
res_imgs_path='./results/pics/'
# load and display instance annotations
dataDir='/home/common/datasets/coco'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
print ("annFile",annFile)
coco=COCO(annFile)


"""
画框函数
"""
def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline = color)


"""
区分不同类别框的颜色
"""
def id_to_color(id):
    #id=id & 63
    num=id+1
    R=(num%2)*10+(num>>1)%2
    G=((num>>1)%2)*10+(num>>1)%2
    B=((num>>1)%2)*10+(num>>1)%2
    R=(id%7)*13+R*4
    G=(id%8)*18+G*6
    B=(id%5)*17+B*9
    return R,G,B


with open(classes_path) as f:
    obj_list = f.readlines()
    obj_list = [x.strip() for x in obj_list]
    
coco_ids= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
        59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
        89, 90]


"""
图片结果绘制
"""
with open(res_dir+"cocoapi_results.json") as rf:
    rf_list=rf.readlines()
    rf_list=[x.strip() for x in rf_list]
    
    with open(img_list_dir+"5k.txt") as f:
        total_img_list = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        total_img_list = [x.strip() for x in total_img_list]
        total_num_t_img = len(total_img_list)
        print("number of images in 5k list: ", total_num_t_img)
        gt_num = 0

        for image_path in total_img_list:
            gt_num += 1
            print(image_path)
            img=Image.open(image_path)
            draw =ImageDraw.Draw(img)
            image_name=int(image_path[50:56])
            print("image_name:",image_name)
            
            #draw GT bbox,class name,score
            imgIds = coco.getImgIds(imgIds = [image_name])
            annIds = coco.getAnnIds(imgIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
#             coco.showAnns(anns)
#             plt.show()
            for n in range(len(anns)):
                print (n)
                x, y, w, h = anns[n]['bbox']
                x, y, w, h = int(x), int(y), int(w), int(h)
                cat=anns[n]['category_id']
                print("gt-obj:",obj_list[coco_ids.index(cat)])
                print(cat,x, y, w, h)
                draw_rectangle(draw, ((x, y), (x + w, y + h)), color=(0,255,0), width=outline_width)
                draw.text((x, y-offset_y), obj_list[coco_ids.index(cat)], font=setFont,fill=(0,255,0), width= 0.3)
            
            
            rf_num=0
            for rf_dict in rf_list:
                rf_dict=ast.literal_eval(rf_dict)
                
                rf_id=rf_dict['image_id']
                if image_name==rf_id:
                    rf_num+=1
#                   print("image_name==rf_id",rf_id)
#                   print ("rf_index=",rf_num)
                    x, y, w, h = rf_dict['bbox']
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    print(x, y, w, h)
                    obj_name=obj_list[coco_ids.index(rf_dict['category_id'])]
                    print_content=obj_name+" conf:"+str(round(rf_dict['score'],3))
                    #outline_width = int(x*y/2000)
                    outline_width=4
                    outline_color = id_to_color(rf_dict['category_id'])
                    #draw_rectangle(draw, ((x, y), (x + w, y + h)), color=outline_color, width=outline_width)
                    setFont= ImageFont.truetype("./font/FiraMono-Medium.otf", 25, encoding="unic")
                    offset_y=30
                    #draw.text((x, y-offset_y), obj_name, font=setFont,fill=id_to_color(rf_dict['category_id']), width= 0.5)
                    draw_rectangle(draw, ((x, y), (x + w, y + h)), color=(255,0,0), width=outline_width)
                    draw.text((x, y-offset_y), print_content, font=setFont,fill=(255,0,0), width= 0.3)
            img.save(res_imgs_path+str(image_name).zfill(6)+'.jpg')
            plt.imshow(img)
            plt.show()
        print("number of images with gt in 5k list: ", gt_num)
    

