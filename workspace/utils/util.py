#!/usr/bin/env python
#-*-coding=utf-8-*-

import numpy as np
import os

import pandas as pd
import zipfile


# source_dir = '/home/mengweiliang/lzh/SqueezeSeg/scripts/log/answers/'
# output_filename = 'answers.zip'

class OutputData(object):
	"""docstring for OutputData"""
	def __init__(self, arg):
		super(OutputData, self).__init__()
		self.arg = arg

	def zip_answers(self, source_dir, output_filename):
		zipf = zipfile.ZipFile(output_filename, 'w')
		pre_len = len(os.path.dirname(source_dir))

		for parent, dirnames, filenames in os.walk(source_dir):
			for filename in filenames:
				pathfile = os.path.join('./', filename)
				arcname = pathfile[pre_len:].strip(os.path.sep)

				zipf.write(pathfile, arcname)
		zipf.close()


		

class InputData(object):
	"""docstring for InputData"""
	def __init__(self, arg):
		super(InputData, self).__init__()
		self.arg = arg
	@property
	def rootPath(self):
		return self._rootPath
	
	@rootPath.setter
	def rootPath(self, path):
		self._rootPath = path

	@property
	def savePath(self):
		if self.rootPath == "":
			self.rootPath = '.'
		return self.rootPath + '/npy/'


	def _cover_csv_to_np(self, file_name, savecsv=False, 
		pts_dir_name='pts', 
		intensity_dir_name='intensity',
		category_dir_name='category'):

		# path
		root_path = self.rootPath

		pts_path = os.path.join(root_path, pts_dir_name, file_name)
		intensity_path = os.path.join(root_path, intensity_dir_name, file_name)
		category_path = os.path.join(root_path, category_dir_name, file_name)

		# pandas
		pts = pd.read_csv(pts_path, header=None)
		intensity = pd.read_csv(intensity_path, header=None)

		if os.path.exists(category_path):
			category = pd.read_csv(category_path, header=None)
		else:
			category = pd.DataFrame(np.zeros(np.shape(intensity), np.float32))

		cantact = pd.concat([pts, intensity, category], axis=1)

		data = pd.DataFrame(concat)
		data.columns = ['x', 'y', 'z', 'i', 'c']
		data.insert(4, 'r', 0)

		data['r'] = np.sqrt(data['x'] ** 2 + data['y'] ** 2 + data['z'] ** 2)

		if savecsv:
			csv_path = os.path.join(root_path, 'csv', file_name)
			data.to_csv(csv_path, index=False, header=False)

		return data.values

	def get_point(self, theta, phi):
        
        # image x(height) * y(width) 2d
        # 向下取整
        x = int((theta - (-16)) / (32.0 / 64))
        y = int((phi - 45.0) / (90.0 / 512))
    
        # 严防越界
        x = (x > 63) and 63 or x
        y = (y > 511) and 511 or y
        
        return x, y

    def get_thetaphi(self, x, y, z):
        theta, phi = self.get_degree(x, y, z)
        
        return self.get_point(theta, phi)

    def get_point_theta(self, x, y, z):
        theta, phi = self.get_degree(x, y, z)
        
        return self.get_point(theta, phi)[0]
    
    def get_point_phi(self, x, y, z):
        theta, phi = self.get_degree(x, y, z)
        
        return self.get_point(theta, phi)[1]
    

    def isempty(self, x):
        if (x==[0, 0, 0]).all():
            return True
        else:
            return False

    # 加载所有的文件名
    def load_file_names(self):
        assert self.rootPath != "", "root path is empty"
        rootname = self.rootPath + '/pts'

        return self.load_subnames(rootname)
    
    
    def load_subnames(self, rootpath):
        
        result = []
        ext = ['csv', 'npy']
    
        files = self._filenames(rootpath)
        for file in files:
            if file.endswith(tuple(ext)):
                result.append(file)
    
        return result

       # 所有子文件
    def _filenames(self, filedir):
        result = []
        for root, dirs, files in os.walk(filedir):
            # print "root: {0}".format(root)
            # print "dirs: {0}".format(dirs)
            # print "files: {0}".format(files)
            result = files
        return result
    
    # 统计标记数量
    def _array_flag_count(self, array, flag):
        count = 0
        for num in array:
            if num == flag:
                count += 1
        return count


    def _generate_image_to_np(self, values, debug=False):

    	data = values

    	x = [data[i][0] for i in range(len(data[:, 0]))]
        y = [data[i][1] for i in range(len(data[:, 0]))]
        z = [data[i][2] for i in range(len(data[:, 0]))]
        intensity = [data[i][3] for i in range(len(data[:, 0]))]
        distance = [data[i][4] for i in range(len(data[:, 0]))]
        label = [data[i][5] for i in range(len(data[:, 0]))]
    
        thetaPt = [self.get_point_theta(data[i][0], data[i][1], data[i][2]) for i in range(len(data[:, 0]))] # x
        phiPt = [self.get_point_phi(data[i][0], data[i][1], data[i][2]) for i in range(len(data[:, 0]))] # y
        
        # 生成数据 phi * theta * [x, y, z, i, r, c]
        image = np.zeros((64, 512, 6), dtype=np.float16)
        
        def store_image(index):
            # print (theta[index], phi[index])
            
            image[thetaPt[index], phiPt[index], 0:3] = [x[index], y[index], z[index]]
            image[thetaPt[index], phiPt[index], 3] = intensity[index]
            image[thetaPt[index], phiPt[index], 4] = distance[index]
            image[thetaPt[index], phiPt[index], 5] = label[index]
        
        for i in range(len(x)):
            if x[i] < 0.5: continue # 前向
            if abs(y[i]) < 0.5: continue
            
            if self.isempty(image[thetaPt[i], phiPt[i], 0:3]):
                store_image(i)
            elif label[i] == image[thetaPt[i], phiPt[i], 5]:
                if distance[i] < image[thetaPt[i], phiPt[i], 4]:
                    image[thetaPt[i], phiPt[i], 4] = distance[i]
            elif image[thetaPt[i], phiPt[i], 5] == 0 and label[i] != 0:
                store_image(i)
            else:
                if distance[i] < image[thetaPt[i], phiPt[i], 4]:
                    store_image(i)
        if debug:
            
            # print theta, phi
            start = time.time()
            for i in range(len(x)):
                # print x[i], y[i], z[i], intensity[i], distance[i], label[i]
                value = x[i]
    
            print time.time() - start
    
            start = time.time()
            for i in range(len(x)):
                # print data[i]
                value = data[i]
    
            print time.time() - start
            
            print source.values
            print 'type: %s' % type(source.values)
            print np.shape(source.values)
            print np.shape(image)
        
        return image
    	pass

    def cover_csv_to_npy(filename, debug=False):

    	data = self._cover_csv_to_np(filename)
    	print(np.shape(data))

    	format_data = self._generate_image_to_np(data)
    	
    	return format_data

