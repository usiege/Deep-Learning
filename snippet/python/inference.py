#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author:charles
# datetime:2019/5/8 下午4:01
# software:PyCharm


import tensorflow as tf
import numpy as np
import math
import random
import cv2
import os
import sys
import json
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--net', default="mrcnn", help='mrcnn or ssd')
parser.add_argument('--path', default="train2", help='data_path')
parser.add_argument('--mode', default="pretrain", help='pretrain or inference')
FLAGS = parser.parse_args()

# mrcnn
import MRCNN.mrcnn.model as modellib
import MRCNN.coco as coco
COCO_DIR = '/data1/coco2014'
class_names =  ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck','boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse','sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie','suitcase',
                'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard','tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple','sandwich',
                'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch','potted plant',
                'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone','microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear','hair drier',
                'toothbrush']

KAGGLE_DIR = '/data1/lzh/product'
#######################################################################################
# ssd

from ssd.nets import ssd_vgg_300, ssd_common, np_methods
from ssd.preprocessing import ssd_vgg_preprocessing
from ssd.notebooks import visualization

#######################################################################################


import data

def test(net='ssd'):
    
    if net == 'ssd':
        print('ssd')
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        slim = tf.contrib.slim
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        isess = tf.InteractiveSession(config=config)
        
        # Input placeholder.
        net_shape = (300, 300)
        data_format = 'NHWC'
        img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        
        # Evaluation pre-processing: resize to SSD net shape.
        image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        image_4d = tf.expand_dims(image_pre, 0)
        
        # Define the SSD model.
        reuse = True if 'ssd_net' in locals() else None
        ssd_net = ssd_vgg_300.SSDNet()
        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
            predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

        # Restore SSD model.
        ckpt_filename = './ssd/checkpoints/ssd_300_vgg.ckpt'
        # ckpt_filename = './ssd/checkpoints/ssd_512_vgg.ckpt'
        # ckpt_filename = './ssd/checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
        isess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(isess, ckpt_filename)
        
        # SSD default anchor boxes.
        ssd_anchors = ssd_net.anchors(net_shape)
        
        bbox_count = 0
        img_count = 0
        imgs = data.load()
        # imgs = ["/data2/products/imat_product_val_20190402/529c41d947bb2fde0e9e0bf770dd4bc0.jpg"]
        for img in imgs:
            img_count += 1
            print(img)
            print(img_count)
            start = time.time()
            image = mpimg.imread(img)
            
            # Run SSD network.
            rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                                      feed_dict={img_input: image})
            # Get classes and bboxes from the net outputs.
            select_threshold = 0.5
            nms_threshold = .45
            rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, ssd_anchors,
                select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

            rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
            rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
            rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
            # Resize bboxes to original image shape. Note: useless for Resize.WARP!
            rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
            
            print(rclasses)
            print(rscores)
            print(rbboxes)
            
            if len(rclasses) > 0:
                bbox_count += 1
                # visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
        
            print(time.time()-start)
            print(bbox_count)

    elif net == 'mrcnn':
    
        print("maskrcnn")

        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        
        COCO = True
        if COCO:
            ROOT_DIR = COCO_DIR
            MODEL_DIR = os.path.join(ROOT_DIR, "logs")
            MODEL_WEIGHT_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')
            
            config = coco.CocoConfig()
            config.IMAGES_PER_GPU = 1
            config.BATCH_SIZE = 1
            config.display()
        else:
            ROOT_DIR = KAGGLE_DIR
            MODEL_DIR = os.path.join(ROOT_DIR, "logs")
            MODEL_WEIGHT_ROOT_PATH = "/data1/lzh/product/logs/kaggle20190525T1910/"
            MODEL_WEIGHT_PATH = MODEL_WEIGHT_ROOT_PATH + "mask_rcnn_kaggle_0100.h5"
    
            config = data.KaggleConfig()
            config.IMAGES_PER_GPU = 1
            config.BATCH_SIZE = 1
            config.display()
    
        with tf.device("/gpu: 2"):
            model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                      config=config)
            weights_path = MODEL_WEIGHT_PATH
            print("Loading weights ", weights_path)
            model.load_weights(weights_path, by_name=True)
            
            res = []
            cur_dir = FLAGS.path
            # cur_dir = data.TEST_DIR
            imgs = data.load(sub_dir=cur_dir)
            for img in imgs:
                print(img)
                mode = FLAGS.mode
                if mode == "pretrain":
                    img_name = img.split('/')[-1][:-4]
                    save_path = os.path.join(data.res_path(cur_dir), "{}.json".format(img_name))
    
                    if os.path.exists(save_path):
                        print(img_name)
                        continue
                        
                    image = cv2.imread(img)
                    results = model.detect([image], verbose=1)[0]
    
                    name = os.path.basename(img)
                    rois = results['rois']
                    classname = [class_names[int(i)] for i in results['class_ids']]
                    score = results['scores']
    
                    dic = {'rois': rois.tolist(),
                           'classnames': classname,
                           'scores': score.tolist()}
    
                    with open(save_path, 'w') as f:
                        js = json.dumps(dic, sort_keys=True, indent=4, separators=(',', ':'))
                        f.write(js)
                        f.close()
                        
                elif mode == 'inference':
                    img_name = img.split('/')[-1]
                    inference_json = os.path.join(MODEL_WEIGHT_ROOT_PATH, f"{cur_dir}.csv")
                    
                    image = cv2.imread(img)
                    results = model.detect([image], verbose=1)[0]

                    name = os.path.basename(img)
                    rois = results['rois']
                    classname = results['class_ids']
                    score = results['scores']
                    
                    print(classname)
                    
                    if len(classname) >= 3:
                        classname = classname[:3]
                    elif len(classname) == 2:
                        classname = [classname[0], classname[1], classname[1]]
                    elif len(classname) == 1:
                        classname = [classname[0], classname[0], classname[0]]
                    else:
                        classname = [0] * 3
                    
                    with open(inference_json, 'w', newline='') as f:
                        writer = csv.writer(f, dialect='excel')
                        class_str = ""
                        for c in classname:
                            class_str += f"{c} "
                        something = [img_name, class_str]
                        print(something)
                        writer.writerow(something)
                        
                    
    # elif net == 'ssd':
    #     print('ssd')

    

if __name__ == '__main__':
    net = FLAGS.net
    test(net=net)
    
