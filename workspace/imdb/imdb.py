#!/usr/bin/env python
#-*-coding=utf-8-*-


import os
import random
import numpy as np

from utils import *

class imdb(object):
	"""docstring for imdb"""

	TOTAL_DATA_COUNT = 50000
	TRAIN_DATA_COUNT = 40000


	def __init__(self, arg):
		super(imdb, self).__init__()
		self.arg = arg
			
		self._name = name

		self._train_index_list = [i+1 for i in np.arange(self.TRAIN_DATA_COUNT)] 
		self._evl_index_list = [i+1 for i in range(self.TRAIN_DATA_COUNT, self.TOTAL_DATA_COUNT)]

		self._cur_idx = 0
		
	@property
	def name(self):
		return self._name
	
	@property
	def data_root_path(self):
		return self._data_root_path

	@property
	def alibaba_data_path(self):
		return os.path.join(self._data_root_path, 'npy')
	

	def _alibaba_pointcloud_path_at(idx):
		pointcloud_path = self.alibaba_data_path + '/channelVELO_TOP_0000_%05d.npy' % int(idx)

		assert os.path.exists(pointcloud_path), 'File does not exist: {}'.format(pointcloud_path)

		return pointcloud_path

	def _load_eval_set_idx(self):
		# path_pre = 'channelVELO_TOP_0000_'
		idxs = [i for i in range(self.TRAIN_DATA_COUNT + 1, self.TOTAL_DATA_COUNT + 1)]

		return idxs

	def read_batch(self, shuffle=True):

		mc = self.mc

		if shuffle:
			if self._cur_idx + mc.BATCH_SIZE >= self.TRAIN_DATA_COUNT:
				self._train_index_list = [self._train_index_list[i] \
				for i in np.random.permutation(np.arange(len(self._train_index_list)))]
				self._cur_idx = 1

			batch_idx = self._train_index_list[self._cur_idx: self._cur_idx + mc.BATCH_SIZE]
			self._cur_idx += mc.BATCH_SIZE
		else:
			eval_count = self.TOTAL_DATA_COUNT - self.TRAIN_DATA_COUNT
			if self._cur_idx + mc.BATCH_SIZE >= eval_count:
				self._evl_index_list = [self._evl_index_list[i] \
				for i in np.random.permutation(np.arange(eval_count))]
				self._cur_idx = 1

			batch_idx = self._evl_index_list[self._cur_idx: self._cur_idx + mc.BATCH_SIZE]
			self._cur_idx += mc.BATCH_SIZE


		 # lidar input: batch * height * width * 5
        lidar_per_batch = []
        # lidar mask, 0 for missing data and 1 otherwise: batch * height * width * 1
        lidar_mask_per_batch = []
        # point-wise labels : batch * height * width
        label_per_batch = []
        # loss weights for different classes: batch * height * width 
        weight_per_batch = []


        for idx in batch_idx:
        	record = np.load(self._alibaba_pointcloud_path_at(idx)\
        		.astype(np.float32, copy=False))

        	lidar = record[:, :, :5] # x, y, z, intensity, range

        	# 512 64
        	lidar_mask = np.reshape((lidar[:, :, 4] > 0),\
        		[mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1])

        	# normalize
        	lidar = (lidar - mc.INPUT_MEAN) / mc.INPUT_STD

        	label = record[:, :, 5]
        	weight = np.zeros(label.shape)

        	for l in range(mc.NUM_CLASS):
        		weight[label==l] = mc.CLS_LOSS_WEIGHT[int(l)]

        	lidar_per_batch.append(lidar)
            lidar_mask_per_batch.append(lidar_mask)
            label_per_batch.append(label)
            weight_per_batch.append(weight)

        return np.array(lidar_per_batch), np.array(lidar_mask_per_batch), \
               np.array(label_per_batch), np.array(weight_per_batch)

