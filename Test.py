import numpy as np
import tensorflow as tf
import Model
import os
import UTILS
import time

meta_data_path = r'./DB/BenchmarkMetadataRaw'
#test_data_path = r'./DB/ValidationNoisyBlocksRaw.mat'
test_data_path = r'./DB/BenchmarkNoisyBlocksRaw.mat'
save_mat_file = 'Results'

gt_path = r'./DB/SIDD_Medium_RawData/Data/0179_008_S6_03200_00800_5500_L/0179_GT_RAW_011.MAT'
noisy_path = r'./DB/SIDD_Medium_RawData/Data/0179_008_S6_03200_00800_5500_L/0179_NOISY_RAW_010.MAT'

# Make graph
model = Model.Model(1e-4)
tf_sess = tf.Session()
tf_saver = tf.train.Saver()
tf_coord = tf.train.Coordinator()
tf_init = tf.global_variables_initializer()


TestData, Param_list = UTILS.TestDataLoad(meta_data_path, test_data_path)

NumTestImg, NumTestBlock, Height, Width = TestData.shape
# Run
ResultData = []
with tf_sess as sess:
    tf_init = tf.global_variables_initializer()
    if model.load(sess, tf_saver,'checkpoint'):
        print('Data loaded')
    else:
        print('Cannot load data!')
        exit(1)
    #start_time = time.time()
    for img_idx in range(NumTestImg):
        print('%d / %d' % (img_idx , NumTestImg))
        curr_data = TestData[img_idx]
        CurrParam = Param_list[img_idx]
        
        if CurrParam[1] == 1:
            curr_data = curr_data[:,:,::-1]
        elif CurrParam[1] == 2:
            curr_data = curr_data[:,::-1,:]
        elif CurrParam[1] == 3:
            curr_data = curr_data[:,::-1,::-1]

        rgb_data = np.empty([NumTestBlock, Height//2, Width//2, 4])
        rgb_data[:,:,:,0] = curr_data[:,0::2, 0::2]
        rgb_data[:,:,:,1] = curr_data[:,0::2, 1::2]
        rgb_data[:,:,:,2] = curr_data[:,1::2, 0::2]
        rgb_data[:,:,:,3] = curr_data[:,1::2, 1::2]
        
        if CurrParam[0] == 0:
            tmpParam = np.array([1, 0, 0, 0, 0, CurrParam[2]/1000, CurrParam[3]/1000])
        elif CurrParam[0] == 1:
            tmpParam = np.array([0, 1, 0, 0, 0, CurrParam[2]/1000, CurrParam[3]/1000])
        elif CurrParam[0] == 2:
            tmpParam = np.array([0, 0, 1, 0, 0, CurrParam[2]/1000, CurrParam[3]/1000])
        elif CurrParam[0] == 3:
            tmpParam = np.array([0, 0, 0, 1, 0, CurrParam[2]/1000, CurrParam[3]/1000])
        elif CurrParam[0] == 4:
            tmpParam = np.array([0, 0, 0, 0, 1, CurrParam[2]/1000, CurrParam[3]/1000])

        tmpParam = np.expand_dims(tmpParam, axis=0)
        tmpParam = np.repeat(tmpParam, NumTestBlock, axis=0)
        result_rgb = sess.run(model.clean_img, {model.input:rgb_data, model.parameters:tmpParam, model.is_training:False})
        result_rgb[result_rgb < 0] = 0
        result_rgb[result_rgb > 1] = 1
        result_img = np.empty([NumTestBlock, Height, Width])
        result_img[:,0::2, 0::2] = result_rgb[:,:,:,0]
        result_img[:,0::2, 1::2] = result_rgb[:,:,:,1]
        result_img[:,1::2, 0::2] = result_rgb[:,:,:,2]
        result_img[:,1::2, 1::2] = result_rgb[:,:,:,3]

        if CurrParam[1] == 1:
            result_img = result_img[:,:,::-1]
        elif CurrParam[1] == 2:
            result_img = result_img[:,::-1,:]
        elif CurrParam[1] == 3:
            result_img = result_img[:,::-1,::-1]
        
        ResultData.append(result_img)
    #end_time = time.time()

    #Time test
    start_time = time.time()
    curr_data = UTILS.bayer_read(noisy_path)
    curr_data = curr_data[:1000,:1000]
    Height, Width = curr_data.shape
    rgb_data = np.empty([Height//2, Width//2, 4])
    rgb_data[:,:,0] = curr_data[0::2, 0::2]
    rgb_data[:,:,1] = curr_data[0::2, 1::2]
    rgb_data[:,:,2] = curr_data[1::2, 0::2]
    rgb_data[:,:,3] = curr_data[1::2, 1::2]
    
    gt_data = UTILS.bayer_read(gt_path)
    gt_data=gt_data[:1000,:1000]
    gt_rgb_data = np.empty([Height//2, Width//2, 4])
    gt_rgb_data[:,:,0] = gt_data[0::2, 0::2]
    gt_rgb_data[:,:,1] = gt_data[0::2, 1::2]
    gt_rgb_data[:,:,2] = gt_data[1::2, 0::2]
    gt_rgb_data[:,:,3] = gt_data[1::2, 1::2]

    parameter = [1, 0, 0, 0, 0, 3200/1000, 800/1000]
    [result_rgb] = sess.run([model.clean_img], {model.input:[rgb_data], model.parameters:[parameter], model.is_training:False})
    end_time = time.time()
    mega_pertime = (end_time - start_time)
    print('Runtime per mega pixel [s] : %.6f' % mega_pertime)
    ResultData = np.asarray(ResultData).astype(np.float32)
    UTILS.SaveMat(ResultData)
    with open('readme.txt', 'w') as f:
        f.write('Runtime per mega pixel [s] : %.6f\n' % mega_pertime)
        f.write('CPU[1] / GPU[0] : 0\n')
        f.write('Use of Metadata [1] / No use of metadata [0] : 1\n')
        f.write('Other: -\n')

    os.rename('results.mat', 'results.MAT')


    