#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author:charles
# datetime:2019/5/8 下午5:14
# software:PyCharm

import os
import glob
import sys

import cv2
import json
import csv

import numpy as np

sys.path.append("./MRCNN")
from MRCNN.mrcnn.config import Config
from MRCNN.mrcnn import utils

# dataset from kaggle
DATASET_ROOT = "/data2/products"

VAL_DIR = "imat_product_val_20190402"
TEST_DIR = "imat_product_test_20190402"
TRAIN_DIR = "imat_product_train_20190409"

MASKRCNN_RES_ROOT = os.path.join(DATASET_ROOT, "mrcnn_res")

TRAIN_LABELS_CSV = os.path.join(DATASET_ROOT, 'train_labels.csv')
VAL_LABELS_CSV = os.path.join(DATASET_ROOT, 'val_labels.csv')

def res_path(sub_dir, root_dir=""):
    return os.path.join(MASKRCNN_RES_ROOT, sub_dir)

def load(sub_dir=VAL_DIR):
    root_path = os.path.join(DATASET_ROOT, sub_dir)
    jpgs = glob.glob(os.path.join(root_path, "*.jpg"))
    
    return jpgs

# dateset for training
MODEL_WEIGHT_ROOT = "/data1/lzh/product"
COCO_MODEL_WEIGHT_PATH = os.path.join("/data1/coco2014", 'mask_rcnn_coco.h5')

class KaggleConfig(Config):
    
    NAME = "kaggle"
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    
    NUM_CLASSES = 1 + 2019
    
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    
    TRAIN_ROIS_PER_IMAGE = 32
    
    STEPS_PER_EPOCH = 100
    
    VALIDATION_STEPS = 5
    
config = KaggleConfig()
config.display()

class KaggleDataset(utils.Dataset):
    
    SUB_DIR = ""
    
    def load_products(self, sub_dir=""):
        self.SUB_DIR = sub_dir
        # add classes
        for i in range(1, config.NUM_CLASSES):
            class_id = i
            class_name = str(i)
            self.add_class("kaggle", class_id=class_id, class_name=class_name)
        
        # image_info
        root_path = os.path.join(DATASET_ROOT, sub_dir)
        jpgs = glob.glob(os.path.join(root_path, "*.jpg"))
        for i in range(len(jpgs)):
            self.add_image("kaggle", image_id=i, path=jpgs[i])
    
    def load_image(self, image_id):
        info = self.image_info[image_id]
        path = info["path"]
        
        image = cv2.imread(path)
        h, w, _ = image.shape
        info['shape'] = (h, w) # row column
        self.image_info[image_id] = info
        
        return image
    
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info["path"]
        image_name = path.split('/')[-1][:-4]

        # load labels
        count = 1
        h, w = info['shape']
        labels = {}
        train_file = open(TRAIN_LABELS_CSV, 'r')
        train_dic = csv.reader(train_file)
        val_file = open(VAL_LABELS_CSV, 'r')
        val_dic = csv.reader(val_file)
        for item in train_dic:
            labels[item[0]] = item[1]
        for item in val_dic:
            labels[item[0]] = item[1]
        
        # load bbox used coco weight model
        json_path = os.path.join(MASKRCNN_RES_ROOT, self.SUB_DIR, f"{image_name}.json")
        with open(json_path, 'r') as f:
            jsobj = json.loads(f.read())
            classnames = jsobj['classnames']
            rois = jsobj['rois'] # y1, x1, y2, x2
            scores = ['scores']
            
            useful_bbox = []
            for i in range(len(classnames)):
                if not classnames[i] == 'person':
                    useful_bbox.append(rois[i])
            
            if len(useful_bbox) > 0:
                y1s = []; x1s = []; y2s = []; x2s = []
                for i in range(len(useful_bbox)):
                    ub = useful_bbox[i]
                    y1s.append(ub[0])
                    x1s.append(ub[1])
                    y2s.append(ub[2])
                    x2s.append(ub[3])
                if len(useful_bbox) > 1:
                    y1, x1, y2, x2 = min(y1s), min(x1s), max(y2s), max(x2s)
                else:
                    y1, x1, y2, x2 = useful_bbox[0]
            else:
                y1, x1, y2, x2 = 0, 0, h, w
                
        # mask
        mask = np.ones([h, w, count], dtype=np.uint8)
        mask[y1:y2+1, x1:x2+1, :] = 0
        mask = np.logical_not(mask).astype(np.bool)
        
        # class
        class_name = labels[f"{image_name}.jpg"]
        class_ids = np.array([int(class_name)]).astype(np.int32)
    
        return mask, class_ids
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    