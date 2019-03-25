import scipy.io
import mat4py
import os
import h5py
import numpy as np
import Config

DictModelMode = {'S6':0, 'GP':2, 'N6':2, 'G4':2, 'IP':1}
DictModelIdx = {'S6':0, 'GP':1, 'N6':2, 'G4':3, 'IP':4}

def meta_read(file_path):
    # Read MetaData
    mat = scipy.io.loadmat(file_path)
    file_name = ''.join(mat['metadata']['Filename'][0][0])
    file_name_split = file_name.replace('/','\\').split('\\')
    infomation = file_name_split[-3].split('_')
    scene_instance_number       = int(infomation[0])
    scene_number                = int(infomation[1])
    smartphone_code             = infomation[2]
    ISO_level                   = int(infomation[3])
    shutter_speed               = int(infomation[4])
    illuminant_temperature      = int(infomation[5])
    illuminant_brightness_code  = infomation[6]

    return smartphone_code, ISO_level, shutter_speed

def bayer_read(file_path):
    arrays = {}
    with h5py.File(file_path, 'r') as f:
        return (np.array(f.get('x'))).T

def get_training_file_list_by_metafile():
    input_data_names = []
    folder_path = Config.Training_File_Path
    sub_folder_path = os.listdir(folder_path)
    
    for sub_folder_name in sub_folder_path:
        files_in_folder = os.listdir(folder_path + '/' + sub_folder_name)
    
        for data_f in files_in_folder:
            if data_f.find('METADATA') != -1:
                input_data_names.append(folder_path + '/' + sub_folder_name + '/' + data_f)
    
    print('#Training_files: %d' % (len(input_data_names)))
    return input_data_names



def get_sample_from_file(file_name, patch_size):

    gt_path = file_name.replace('METADATA', 'GT')
    noisy_path = file_name.replace('METADATA', 'NOISY')
    smartphone_code, ISO_level, shutter_speed = meta_read(file_name)
    mode = DictModelMode.get(smartphone_code)

    gt = bayer_read(gt_path)
    noisy = bayer_read(noisy_path)

    #Mirror X / Y to make RAW Gr first.
    if mode == 1:
        gt = gt[:,::-1]
        noisy = noisy[:,::-1]
    elif mode == 2:
        gt = gt[::-1,:]
        noisy = noisy[::-1,:]
    elif mode == 3:
        gt = gt[::-1,::-1]
        noisy = noisy[::-1,::-1]


    h, w = gt.shape
    
    #Random sampling
    s_x = (np.random.random_integers(0, w - patch_size)//2)*2
    s_y = (np.random.random_integers(0, h - patch_size)//2)*2
    e_x = s_x + patch_size
    e_y = s_y + patch_size

    gt = gt[s_y:e_y, s_x:e_x]
    noisy = noisy[s_y:e_y, s_x:e_x]
    mode = DictModelMode.get(smartphone_code)
    if mode == 0:
        params=[1, 0, 0, 0, 0, ISO_level, shutter_speed]
    elif mode == 1:
        params=[0, 1, 0, 0, 0, ISO_level, shutter_speed]
    elif mode == 2:
        params=[0, 0, 1, 0, 0, ISO_level, shutter_speed]
    elif mode == 3:
        params=[0, 0, 0, 1, 0, ISO_level, shutter_speed]
    elif mode == 4:
        params=[0, 0, 0, 0, 1, ISO_level, shutter_speed]

    
    gt_4ch = np.empty([patch_size//2, patch_size//2, 4])
    noisy_4ch = np.empty([patch_size//2, patch_size//2, 4])
    
    gt_4ch[:,:,0] = gt[0::2, 0::2]
    gt_4ch[:,:,1] = gt[0::2, 1::2]
    gt_4ch[:,:,2] = gt[1::2, 0::2]
    gt_4ch[:,:,3] = gt[1::2, 1::2]
    
    noisy_4ch[:,:,0] = noisy[0::2, 0::2]
    noisy_4ch[:,:,1] = noisy[0::2, 1::2]
    noisy_4ch[:,:,2] = noisy[1::2, 0::2]
    noisy_4ch[:,:,3] = noisy[1::2, 1::2]


    return noisy_4ch, gt_4ch, params



def TestDataLoad(MetaData_path, TestData_path):

    # Read test data
    TestData = scipy.io.loadmat(TestData_path)
    TestData = np.array(TestData.get(TestData_path.split('/')[-1].split('.')[0]))
    n, b, h, w = TestData.shape
    Param_list = []
    # Read parameter    
    for i in range(n):
        mat = scipy.io.loadmat(MetaData_path + ('/Metadata_%02d.MAT' % (i+1)))
        file_name = ''.join(mat['metadata']['Filename'][0][0])
        file_name_split = file_name.replace('/','\\').split('\\')
        infomation = file_name_split[-3].split('_')
    
        smartphone_code             = infomation[2]
        ISO_level                   = int(infomation[3])
        shutter_speed               = int(infomation[4])
        Param_list.append([DictModelIdx[smartphone_code], DictModelMode[smartphone_code], ISO_level, shutter_speed])

    
    return TestData, Param_list


def SaveMat(data):
    scipy.io.savemat('Results',{'results':data})