#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author:charles
# datetime:2019/5/8 下午5:11
# software:PyCharm

from data import *
from MRCNN.mrcnn.config import Config
from MRCNN.mrcnn import utils
from MRCNN.mrcnn import visualize
from MRCNN.mrcnn.model import log
import MRCNN.mrcnn.model as modellib

import tensorflow as tf

def train():
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
    slim = tf.contrib.slim
    gpu_options = tf.GPUOptions(allow_growth=True)
    tf_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    isess = tf.InteractiveSession(config=tf_config)
    
    dataset_train = KaggleDataset()
    dataset_train.load_products(sub_dir="train1")
    dataset_train.prepare()
    
    dataset_val = KaggleDataset()
    dataset_val.load_products(sub_dir=VAL_DIR)
    dataset_val.prepare()
    
    # model
    model_dir = MODEL_WEIGHT_ROOT + "/logs"
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=model_dir)
    model.load_weights(COCO_MODEL_WEIGHT_PATH, by_name=True, exclude=[
        "mrcnn_class_logits",
        "mrcnn_bbox_fc",
        "mrcnn_bbox",
        "mrcnn_mask"
    ])
    
    # train
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE, epochs=100, layers="heads")


if __name__ == "__main__":
    print("train")
    train()