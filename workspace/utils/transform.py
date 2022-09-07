#!/usr/bin/env python
#-*-coding=utf-8-*-

import utils

## training tools
def transform_training_data_to_npys(root_path):

	print root_path

	transtool = urils.InputData()
	transtool.rootPath = root_path

	pts_files = transtool.load_file_names()

	filesname_savepath = "../../scripts/log/filenames.txt"
    with open(filesname_savepath, 'w') as f:
        for i in range(0, len(ptsfiles)):
            context = ptsfiles[i] + '\n'
            f.write(context)
        f.close()
    
    
    idx = 0
    for file in ptsfiles:
        # print '正在转换 file ：%s  ......' % file
        idx += 1
        if file[-4:] == '.csv':
            prename = 'channelVELO_TOP_0000_%05d' % idx
            npyname = (prename + '.npy')
            
            npypath = trantool.savePath + npyname
            if os.path.exists(npypath):
                continue
            
            # start = time.time()
            data = trantool.cover_csv_to_nzero(file)
            formatdata = trantool.generate_image_np(data, debug=False)
            np.save(npypath, formatdata)
            
            # if np.shape(formatdata) == (64, 512, 6):
            #     print '%s 已生成' % npypath
            #     print '耗时：%s ' % time.time() - start


## testing tools

# transform testing data
def transform_test_data(rootpath=""):
    
    print "test root path is {0}".format(rootpath)
    
    tools = ct.InputData()
    tools.rootPath = rootpath
    
    test_name_path = rootpath + "/intensity"
    test_file_names = tools.load_subnames(test_name_path)
    
    for index, file_name in enumerate(test_file_names):
        
        if file_name[-4:] == '.csv':
            npypath = tools.savePath + file_name[:-4] + '.npy'
            
            if os.path.exists(npypath):
                continue
            
            data = tools.cover_csv_to_nzero(file_name)
            # formatdata = tools.generate_image_np(data, debug=False)
            result = data.values
            print np.shape(result)
            np.save(npypath, result)
            
            # if np.shape(formatdata) == (64, 512, 6):
            #     print '%s has generated' % (npypath)

def save_names_file(rootpath, save_file):
    
    tools = utils.InputData()
    names = tools.load_subnames(rootpath)
    
    with open(save_file, 'w') as f:
        for name in names:
            context = name + '\n'
            f.write(context)
        f.close()



if __name__ == '__main__':

    path = "/home/mengweiliang/disk15/df314/training"
    transform_training_npy(path)

    # testpath = '/home/mengweiliang/disk15/df314/test'
    # transform_test_data(testpath)





    