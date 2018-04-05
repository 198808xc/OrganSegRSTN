# OrganSegRSTN: an end-to-end coarse-to-fine organ segmentation framework
version 1.0 - Apr 5 2018 - by Qihang Yu, Yuyin Zhou and Lingxi Xie

#### **Qihang Yu and Yuyin Zhou are the main contributors to this repository.**

Yuyin Zhou implemented [the original coarse-to-fine framework](https://github.com/198808xc/OrganSegC2F),
Qihang Yu improved it to allow end-to-end training, and Lingxi Xie later wrapped up these codes for release.

#### If you use our codes, please cite our paper accordingly:

  **Qihang Yu**, Lingxi Xie, Yan Wang, Yuyin Zhou, Elliot K. Fishman, Alan L. Yuille,
    "Recurrent Saliency Transformation Network: Incorporating Multi-Stage Visual Cues for Small Organ Segmentation",
    in IEEE Conference on CVPR, Salt Lake City, Utah, USA, 2018.

https://arxiv.org/abs/1709.04518

###### and possibly, our previous work (the basis of this work):

  **Yuyin Zhou**, Lingxi Xie, Wei Shen, Yan Wang, Elliot K. Fishman, Alan L. Yuille,
    "A Fixed-Point Model for Pancreas Segmentation in Abdominal CT Scans",
    in International Conference on MICCAI, Quebec City, Quebec, Canada, 2017.

https://arxiv.org/abs/1612.08230

All the materials released in this library can **ONLY** be used for **RESEARCH** purposes.

The authors and their institution (JHU/JHMI) preserve the copyright and all legal rights of these codes.

**Before you start, please note that there is a LAZY MODE,
  which allows you to run the entire framework with ONE click.
  Check the contents before Section 4.3 for details.**


## 1. Introduction

OrganSegRSTN is a code package for our paper:
  **Yuyin Zhou**, Lingxi Xie, Wei Shen, Yan Wang, Elliot K. Fishman, Alan L. Yuille,
    "A Fixed-Point Model for Pancreas Segmentation in Abdominal CT Scans",
    in International Conference on MICCAI, Quebec City, Quebec, Canada, 2017.

OrganSegRSTN is a segmentation framework designed for 3D volumes.
    It was originally designed for segmenting abdominal organs in CT scans,
    but we believe that it can also be used for other purposes,
    such as brain tissue segmentation in fMRI-scanned images.

OrganSegRSTN is based on the state-of-the-art deep learning techniques.
    This code package is to be used with CAFFE, a deep learning library.
    We make use of the python interface of CAFFE, named pyCAFFE.

It is highly recommended to use one or more modern GPUs for computation.
    Using CPUs will take at least 50x more time in computation.


## 2. File List

| Folder/File                | Description                                          |
|:-------------------------- |:---------------------------------------------------- |
| `README.txt`               | the README file                                      |
|                            |                                                      |
| **DATA2NPY/**              | codes to transfer the NIH dataset into NPY format    |
| `dicom2npy.py`             | transferring image data (DICOM) into NPY format      |
| `nii2npy.py`               | transferring label data (NII) into NPY format        |
|                            |                                                      |
| **DiceLossLayer/**         | CPU implementation of the Dice loss layer            |
| `dice_loss_layer.hpp`      | the header file                                      |
| `dice_loss_layer.cpp`      | the CPU implementation                               |
|                            |                                                      |
| **OrganSegC2F/**           | primary codes of OrganSegC2F                         |
| `coarse2fine_testing.py`   | the coarse-to-fine testing process                   |
| `coarse_fusion.py`         | the coarse-scaled fusion process                     |
| `coarse_testing.py`        | the coarse-scaled testing process                    |
| `Crop.py`                  | the crop layer (cropping a region from the image)    |
| `Data.py`                  | the data layer                                       |
| `indiv_training.py`        | training the coarse and fine stages individually     |
| `init.py`                  | the initialization functions                         |
| `joint_training.py`        | training the coarse and fine stages jointly          |
| `Uncrop.py`                | the uncrop layer (putting the regional output back)  |
| `oracle_fusion.py`         | the fusion process with oracle information           |
| `oracle_testing.py`        | the testing process with oracle information          |
| `run.sh`                   | the main program to be called in bash shell          |
| `surgery.py`               | the surgery function                                 |
| `utils.py`                 | the common functions                                 |
|                            |                                                      |
| **OrganSegC2F/prototxts**  | primary codes of OrganSegC2F                         |
| `deploy_C3.prototxt`       | the prototxt file for coarse-scaled testing          |
| `deploy_F3.prototxt`       | the prototxt file for fine-scaled testing            |
| `deploy_O3.prototxt`       | the prototxt file for oracle testing                 |
| `training_I3x1.prototxt`   | the prototxt file for individual training (1xLR)     |
| `training_I3x10.prototxt`  | the prototxt file for individual training (10xLR)    |
| `training_J3x1.prototxt`   | the prototxt file for joint training (1xLR)          |
| `training_J3x10.prototxt`  | the prototxt file for joint training (10xLR)         |
| `training_S3x1.prototxt`   | the prototxt file for separate training (1xLR)       |
| `training_S3x10.prototxt`  | the prototxt file for separate training (10xLR)      |

The multiplier (1 or 10) applies to all the trainable layers in the fine stage of the framework.


## 3. Installation


#### 3.1 Prerequisites

###### 3.1.1 Please make sure that your computer is equipped with modern GPUs that support CUDA.
    Without them, you will need 50x more time in both training and testing stages.

###### 3.1.2 Please also make sure that python (we are using 2.7) is installed.


#### 3.2 CAFFE and pyCAFFE

###### 3.2.1 Download a CAFFE library from http://caffe.berkeleyvision.org/ .
    Suppose your CAFFE root directory is $CAFFE_PATH.

###### 3.2.2 Place the files of Dice loss layer at the correct position.
    dice_loss_layer.hpp -> $CAFFE_PATH/include/caffe/layers/
    dice_loss_layer.cpp -> $CAFFE_PATH/src/caffe/layers/

###### 3.2.3 Make CAFFE and pyCAFFE.


