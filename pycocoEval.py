#-*- coding:utf-8 -*-
"""
Created on April, 2019
@authors: Hulking
"""
"""
计算mAP
"""
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab,json
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
def get_img_id(file_name):
    ls = []
    myset = []
    annos = json.load(open(file_name, 'r'))
    for anno in annos:
      ls.append(anno['image_id'])
    myset = {}.fromkeys(ls).keys()
    return myset
def cal_coco_map():
    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[1]
    cocoGt_file = '/home/common/datasets/coco/annotations/instances_val2014.json'
    cocoGt = COCO(cocoGt_file)
    cocoDt_file = './results/cocoapi_results.json'
    imgIds = get_img_id(cocoDt_file)
    print (len(imgIds))
    cocoDt = cocoGt.loadRes(cocoDt_file)
    imgIds = sorted(imgIds)
    imgIds = imgIds[0:5000]
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == '__main__':
    cal_coco_map()