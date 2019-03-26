# DEep Factorized Network (DEFNet) for image denoising
This source and executable code, based on TensorFlow, is to submit to
  [NTIRE 2019 Real Image Denoising Challenge - Track 1: Raw-RGB](https://competitions.codalab.org/competitions/21258)


## Setup
## Requirement
- Tensorflow (1.13.1)
- Numpy (1.16.2)
- h5py (2.9.0)
- mat4py (0.4.2)
- scipy (1.2.1)

## Getting Started

### Training
Our training strategy consists of the following two steps.
1. At the first step, we train the main denoiser. The squared L2 norm of the difference between the denoised image (x0) and the target image is used as loss. This step is designed to reduce the time spent for learning entire network and can be skipped. In our case, this step was learned for three days. 
You can run this step with the following command:
```bash
$ python Train_Step1.py
```

2. The second step aims to train the entire network for the provided Training data set. We utilize the squared L2 norm of difference between the final resultant image (x3) and the target image. This step is the most important step. We performed this step until the ground truth valid set was released.
You can run this step of training with the following command:
```bash
$ python Train_Step2.py
```

### Test
In test step, a mat file of special validset or testset is used as input. The denoised result images and readme file which contains execution time information are created as mat file and txt file, respectively.
You can run this step of training with the following command:
```bash
$ python Test.py
```

## Dataset
This repository contains Validset / Testset and few Training sets. You need to get the full Training data to learn. You can download the full Training data from the link follow:
[NTIRE 2019 Real Image Denoising Challenge - Track 1: Raw-RGB#participate-get-data](https://competitions.codalab.org/competitions/21258#participate-get-data)

>Training dataset path: _./DB/SIDD_Medium_RawData_.

