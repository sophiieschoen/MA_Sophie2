#!/usr/bin/myenv3.9 python #do I have to put here the path of the virtual environment?
# coding: utf-8

# create the directory tree -> just has to be done once?
import subprocess

# clone the repository
subprocess.run(['git', 'clone', 'https://github.com/MIC-DKFZ/nnUNet.git'])

# navigate to the nnUNet directory
subprocess.run(['cd', 'nnUNet'])

# install nnUNet as a Python package
subprocess.run(['pip', 'install', '-e', '.'])

# set env variables for the bash process #just has to be done once?

import os

os.environ['nnUNet_raw'] = "/mnt/nfs-students/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "/mnt/nfs-students/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/mnt/nfs-students/nnUNet_results"

#%%capture
import subprocess
#subprocess.run(["nnUNet_install_pretrained_model_from_zip $PATH_TO_MODEL_FILE"])

import os
import sys
import shutil

import time
import gdown

import json
import pprint
import numpy as np
import pandas as pd

import pydicom
import nibabel as nib
import SimpleITK as sitk

from medpy.metric.binary import dc as dice_coef
from medpy.metric.binary import hd as hausdorff_distance
from medpy.metric.binary import asd as avg_surf_distance
from medpy.filter.binary import largest_connected_component



##import tensorflow as tf # raises error message: Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /share/software/user/open/python/3.9.0/lib:

#print(tf.__version__)


##import keras # raises error message: Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /share/software/user/open/python/3.9.0/lib:


'''
#in colab: 
Python version               :  3.8.16 (default, Dec  7 2022, 01:12:13) 
Numpy version                :  1.21.6
TensorFlow version           :  2.9.2
Keras (stand-alone) version  :  2.9.0

currently in sherlock shell:
Python version               :  3.6.1 (default, Apr 27 2017, 10:57:56) 
Numpy version                :  1.19.5
TensorFlow version           :  2.6.2
Keras (stand-alone) version  :  2.6.0
'''

print("Python version               : ", sys.version.split('\n')[0])
print("Numpy version                : ", np.__version__)
##print("TensorFlow version           : ", tf.__version__) #name tf is not defined
##print("Keras (stand-alone) version  : ", keras.__version__) #name keras is not defined

print("\nThis Colab instance is equipped with a GPU.")

# ----------------------------------------

#everything that has to do with plotting goes here below
import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "png"')

import ipywidgets as ipyw #-> causes error -> do we need?

## ----------------------------------------

# create new colormap appending the alpha channel to the selected one
# (so that we don't get a \"color overlay\" when plotting the segmask superimposed to the CT)
cmap = plt.cm.Reds
my_reds = cmap(np.arange(cmap.N))
my_reds[:,-1] = np.linspace(0, 1, cmap.N)
my_reds = ListedColormap(my_reds)

cmap = plt.cm.Greens
my_greens = cmap(np.arange(cmap.N))
my_greens[:,-1] = np.linspace(0, 1, cmap.N)
my_greens = ListedColormap(my_greens)

cmap = plt.cm.Blues
my_blues = cmap(np.arange(cmap.N))
my_blues[:,-1] = np.linspace(0, 1, cmap.N)
my_blues = ListedColormap(my_blues)

cmap = plt.cm.spring
my_spring = cmap(np.arange(cmap.N))
my_spring[:,-1] = np.linspace(0, 1, cmap.N)
my_spring = ListedColormap(my_spring)
## ----------------------------------------


import seaborn as sns

import torch  #causes error



#python -c 'import torch;print(torch.backends.cudnn.version())'
#python -c 'import torch;print(torch.__version__)'

print(torch.__version__) #output: 1.13.1+cu117


#we are in path directory: /scratch/users/sschoen so can I just do: 

import subprocess

#DATASET CONVERSION: #already done
##subprocess.run(["nnUNet_convert_decathlon_task", "-i", "/scratch/users/sschoen/Data_MA_PET_resampled/nnUNet_raw_data_base/nnUNet_raw_data/Task01_MA", "-p", "3"]) #-p steht für ?
#converted images are saved in /scratch/users/sschoen/Data/nnUNet_raw_data_base/nnUNet_raw_data/Task001_MA

print("start preprocessing done")
#nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

##EXPERIMENT PLANNING AND PREPROCESSING #already done
# populate your folder with preprocessed data. nnUNet_plan_and_preprocess creates subfolders with preprocessed data for the 2D U-Net as well as all applicable 3D U-Nets. It will also create 'plans' files (with the ending.pkl) for the 2D and 3D configurations. These files contain the generated segmentation pipeline configuration and will be read by the nnUNetTrainer (see below). Note that the preprocessed data folder only contains the training cases. The test images are not preprocessed (they are not looked at at all!). Their preprocessing happens on the fly during inference.
subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", "1", "--verify_dataset_integrity"])
#output preproccesed files in:  /scratch/users/sschoen/Data_MA/nnUNet_preprocessed/Task001_MA
##ERROR:GEOMETRY MISMATCH FOUND ?!
print("preprocessing done")

'''
import collections
try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict
    
from array import array
import numpy as np


#create an PKL file in Python #already done

# all files: 'la_002','la_010','la_012','la_013', 'la_015', 'la_029','la_033','la_038','la_040','la_042','la_045','la_047','la_049','la_055','la_057','la_060','la_067','la_071','la_072','la_074'
import pickle
#create list: #each image is one time in validation set and four times in training set/fold
#split_list=[OrderedDict([('train', np.array(['la_003', 'la_004', 'la_005', 'la_009', 'la_010'], dtype='<U6')),('val', np.array(['la_007']dtype='<U6'))]), OrderedDict([('train', np.array(['la_004', 'la_005', 'la_007', 'la_009', 'la_010'], dtype='<U6')),('val', np.array(['la_003']dtype='<U6'))]), OrderedDict([('train', np.array(['la_003', 'la_005', 'la_007', 'la_009', 'la_010'], dtype='<U6')),('val', np.array(['la_004']dtype='<U6'))]),OrderedDict([('train', np.array(['la_003', 'la_004', 'la_005', 'la_007', 'la_010'], dtype='<U6')),('val', np.array(['la_009']dtype='<U6'))]),OrderedDict([('train', np.array(['la_003', 'la_004', 'la_005', 'la_007','la_009'], dtype='<U6')),('val', np.array(['la_010']dtype='<U6'))])]
split_list=[
OrderedDict([('train', np.array(['la_029','la_032','la_033','la_036','la_037','la_038','la_039','la_040','la_041','la_042','la_045','la_047','la_048','la_049','la_055','la_057','la_061','la_067','la_071','la_072','la_074','la_082'], dtype='<U6')), ('val', np.array(['la_002','la_009','la_010','la_012','la_013','la_015'],dtype='<U6'))]), 
OrderedDict([('train', np.array(['la_002','la_009','la_010','la_012','la_013','la_015','la_039','la_040','la_041','la_042','la_045','la_047','la_048','la_049','la_055','la_057','la_061','la_067','la_071','la_072','la_074','la_082'], dtype='<U6')), ('val', np.array(['la_029','la_032','la_033','la_036','la_037','la_038'], dtype='<U6'))]),
OrderedDict([('train', np.array(['la_002','la_009','la_010','la_012','la_013','la_015','la_029','la_032','la_033','la_036','la_037','la_038','la_048','la_049','la_055','la_057','la_061','la_067','la_071','la_072','la_074','la_082'], dtype='<U6')), ('val', np.array(['la_039','la_040','la_041','la_042','la_045','la_047'], dtype='<U6'))]),
OrderedDict([('train', np.array(['la_002','la_009','la_010','la_012','la_013','la_015','la_029','la_032','la_033','la_036','la_037','la_038','la_039','la_040','la_041','la_042','la_045','la_047','la_071','la_072','la_074','la_082'], dtype='<U6')), ('val', np.array(['la_048','la_049','la_055','la_057','la_061','la_067'], dtype='<U6'))]),
OrderedDict([('train', np.array(['la_002','la_009','la_010','la_012','la_013','la_015','la_029','la_032','la_033','la_036','la_037','la_038','la_039','la_040','la_041','la_042','la_045','la_047','la_048','la_049','la_055','la_057','la_061','la_067'], dtype='<U6')), ('val', np.array([,'la_071','la_072','la_074','la_082'], dtype='<U6'))])]

#split with just one training and vaidation image:
#split_list=[OrderedDict([('train', np.array(['la_003'], dtype='<U6')), ('val', np.array(['la_007'], dtype='<U6'))]),OrderedDict([('train', np.array(['la_003'], dtype='<U6')), ('val', np.array(['la_007'], dtype='<U6'))]), OrderedDict([('train', np.array(['la_003'], dtype='<U6')), ('val', np.array(['la_007'], dtype='<U6'))]), OrderedDict([('train', np.array(['la_003'], dtype='<U6')), ('val', np.array(['la_007'], dtype='<U6'))]), OrderedDict([('train', np.array(['la_003'], dtype='<U6')), ('val', np.array(['la_007'], dtype='<U6'))])]

#split_list=[OrderedDict([('train', 'la_003'),('val', 'la_007')]),OrderedDict([('train', 'la_003'),('val', 'la_007')]),OrderedDict([('train', 'la_003'),('val', 'la_007')]),OrderedDict([('train', 'la_003'),('val', 'la_007')]),OrderedDict([('train', 'la_003'),('val', 'la_007')])] 

#data = pickle.load(f)
with open('/scratch/users/sschoen/Data_MA_PET_resampled/nnUNet_preprocessed/Task001_MA/splits_final.pkl', 'wb') as f: #error can not find file -> should get created automatically?
    pickle.dump(split_list, f)
f.close()

with open('/scratch/users/sschoen/Data_MA_PET_resampled/nnUNet_preprocessed/Task001_MA/splits_final.pkl', 'rb') as file:
    my_split = pickle.load(file)
file.close()
print(my_split)
type(my_split)
'''
