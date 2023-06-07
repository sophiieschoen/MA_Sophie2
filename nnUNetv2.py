#!/usr/bin/myenv3.9 python #do I have to put here the path of the virtual environment?
# coding: utf-8

import subprocess
import os

# clone the repository
subprocess.run(['git', 'clone', 'https://github.com/MIC-DKFZ/nnUNet.git'])

# navigate to the nnUNet directory
current_dir = os.getcwd()


print("Current directory:", current_dir)

os.chdir('nnUNet')
current_dir = os.getcwd()

print("Current directory:", current_dir)


# install nnUNet as a Python package
subprocess.run(['pip', 'install', '-e', '.'])

# set env variables for the bash process #just has to be done once?
import os

os.environ['nnUNet_raw'] = "/mnt/nfs-students/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "/mnt/nfs-students/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/mnt/nfs-students/nnUNet_results"
os.environ["nnUNet_n_proc_D"] = "0"
#os.environ["pin_memory"] = "False" -> geht nicht, muss direkt im dataloader angepasst werden

#export nnUNet_raw="/mnt/nfs-students/nnUNet_raw"
#export nnUNet_preprocessed="/mnt/nfs-students/nnUNet_preprocessed"
#export nnUNet_results="/mnt/nfs-students/nnUNet_results"
##%%capture
import subprocess
#subprocess.run(["nnUNet_install_pretrained_model_from_zip $PATH_TO_MODEL_FILE"])

import os
import sys
import shutil

import time
##import gdown

import json
import pprint
import numpy as np
##import pandas as pd

##import pydicom
import nibabel as nib
import SimpleITK as sitk

##from medpy.metric.binary import dc as dice_coef
##from medpy.metric.binary import hd as hausdorff_distance
##from medpy.metric.binary import asd as avg_surf_distance
##from medpy.filter.binary import largest_connected_component



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

##import ipywidgets as ipyw #-> causes error -> do we need?

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


##import seaborn as sns

##import torch  #causes error


#python -c 'import torch;print(torch.backends.cudnn.version())'
#python -c 'import torch;print(torch.__version__)'

##print(torch.__version__) #output: 1.13.1+cu117


#we are in path directory: /scratch/users/sschoen so can I just do: 

import subprocess

#DATASET CONVERSION: #already done
##subprocess.run(["nnUNet_convert_decathlon_task", "-i", "/scratch/users/sschoen/Data_MA_PET_resampled/nnUNet_raw_data_base/nnUNet_raw_data/Task01_MA", "-p", "3"]) #-p steht für ?
#converted images are saved in /scratch/users/sschoen/Data/nnUNet_raw_data_base/nnUNet_raw_data/Task001_MA
current_dir = os.getcwd()
print("Current directory:", current_dir)

import subprocess

'''
# Get the path of the nnUNetv2_plan_and_preprocess script
command = "which nnUNetv2_plan_and_preprocess"
result = subprocess.run(command.split(), stdout=subprocess.PIPE)
nnUNet_path = result.stdout.decode().strip()
print("nnUNet path=",nnUNet_path)

print("start preprocessing")
#in terminal: nnUNetv2_plan_and_preprocess -d 5 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 6 --verify_dataset_integrity
#nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
# populate your folder with preprocessed data. nnUNet_plan_and_preprocess creates subfolders with preprocessed data for the 2D U-Net as well as all applicable 3D U-Nets. It will also create 'plans' files (with the ending.pkl) for the 2D and 3D configurations. These files contain the generated segmentation pipeline configuration and will be read by the nnUNetTrainer (see below). Note that the preprocessed data folder only contains the training cases. The test images are not preprocessed (they are not looked at at all!). Their preprocessing happens on the fly during inference.
'''
print("start preprocessing")
#subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", "8", "--verify_dataset_integrity"])
print("preprocessing done")

'''
process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print(process.stdout.decode())
print(process.stderr.decode())

command = ["nnUNetv2_plan_and_preprocess", "-d", "1", "--verify_dataset_integrity"]
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

while True:
    output = process.stdout.readline()
    if not output and process.poll() is not None:
        break
    if output:
        print(output.strip())

print("preprocessing done")
'''

print ("start training")

#nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD --val --npz
#nnUNetv2_train 7 3d_fullres 4 --npz
#subprocess.run(["nnUNetv2_train","6", "3d_fullres","0", "--npz"]) #fold0
subprocess.run(["nnUNetv2_train","7", "3d_fullres","2", "--npz"]) #fold1
##subprocess.run(["nnUNetv2_train","8", "3d_fullres","2", "--npz"]) #fold2
#subprocess.run(["nnUNetv2_train","5", "3d_fullres","3", "--npz"]) #fold3
#subprocess.run(["nnUNetv2_train","5", "3d_fullres","4", "--npz"]) #fold4

print ("training completed")

#nnUNetv2_find_best_configuration 7 -c 3d_fullres

'''
Once completed, the command will print to your console exactly what commands you need to run to make predictions. It will also create two files in the nnUNet_results/DATASET_NAME folder for you to inspect:

inference_instructions.txt again contains the exact commands you need to use for predictions
inference_information.json can be inspected to see the performance of all configurations and ensembles, as well as the effect of the postprocessing plus some debug information.
'''

#nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
#nnUNetv2_ensemble -i FOLDER1 FOLDER2 ... -o OUTPUT_FOLDER -np NUM_PROCESSES
#nnUNetv2_apply_postprocessing -i FOLDER_WITH_PREDICTIONS -o OUTPUT_FOLDER --pp_pkl_file POSTPROCESSING_FILE -plans_json PLANS_FILE -dataset_json DATASET_JSON_FILE
