# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import SimpleITK as sitk

import os
import nibabel as nib
import numpy as np

print("start normalization")
# Directory containing the input images
img_dir_dwi = "/mnt/nfs-students/DWI_resampled/fixedmedianheader2/"
img_dir_pet = "/mnt/nfs-students/PET_resampled/fixedmedianheader2/"
#img_dir_labels = "/mnt/nfs-students/Labels_resampled/onLAVAN10/"

print("input paths defined")


'''
# for one image:
img_pathN10="Volumes/workfiles/schoensophie/DWI_nii/N10-DWI-800.nii.gz"
img = nib.load(img_pathN10)

# Get the image data and header
img_data = img.get_fdata()
img_header = img.header.copy()

# Normalize the image data
img_data_norm = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

# Create a new image with the normalized data and the original header
img_norm = nib.Nifti1Image(img_data_norm, img.affine, header=img_header)

# Save the normalized image to the Resampled_DWI directory
img_norm_path = os.path.join("Volumes/workfiles/schoensophie/DWI_resampled_normalized/", "N10_normalized.nii.gz")
#os.makedirs(os.path.dirname(img_norm_path), exist_ok=True)
nib.save(img_norm, img_norm_path)

'''        

# Loop over all files in the directory DWI images
for filename in os.listdir(img_dir_dwi):
    if filename.endswith(".nii.gz"):
        # Check if the output file already exists
        output_folder = '/mnt/nfs-students/DWI_resampled_normalized/fixedmedianheader2/'
        output_path = os.path.join(output_folder, filename.replace('_resampled.nii.gz', '_resampled_normalized.nii.gz'))
        if os.path.exists(output_path):
            print("Output file already exists for", filename)
            continue  # Skip this file and move on to the next one
        output_path = os.path.join(output_folder, filename.replace('_resampled.nii.gz', '_resampled_normalized.nii.gz'))
        if os.path.exists(output_path):
            print("Output file already exists for", filename)
            continue  # Skip this file and move on to the next one

        print("next round")
        # Load the image
        print("load DWI image",filename)
        img_path = os.path.join(img_dir_dwi, filename)
        img = nib.load(img_path)

        #print("old_direction:",img.GetDirection())
        #print("old_origin:" ,img.GetOrigin())
        #print("old_spacing:" ,img.GetSpacing())
        #print("old_size:" ,img.GetSize())

        # Get the image data and header
        img_data = img.get_fdata()
        img_header = img.header.copy()

        # Normalize the image data
        img_data_norm = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
        print("DWI image normalized")

        # Create a new image with the normalized data and the original header
        img_norm = nib.Nifti1Image(img_data_norm, img.affine, header=img_header)

        #print("new_direction:",img_norm.GetDirection())
        #print("new_origin:" ,img_norm.GetOrigin())
        #print("new_spacing:" ,img_norm.GetSpacing())
        #print("new_size:" ,img_norm.GetSize())

        # Save the normalized image to the Resampled_DWI directory
        img_norm_path = os.path.join("/mnt/nfs-students/DWI_resampled_normalized/fixedmedianheader2/", filename.replace(".nii.gz", "_normalized.nii.gz"))
        #os.makedirs(os.path.dirname(img_norm_path), exist_ok=True)
        nib.save(img_norm, img_norm_path)
        print("normalized image saved in:",img_norm_path)

        print("DWI normalized of patient:",filename)


# Loop over all files in the directory PET images
for filename in os.listdir(img_dir_pet):
    print("next round with file:",filename)

    if filename.endswith(".nii.gz"):
        # Check if the output file already exists
        output_folder = '/mnt/nfs-students/PET_resampled_normalized/fixedmedianheader2/'
        output_path = os.path.join(output_folder, filename.replace('_resampled.nii.gz', '_resampled_normalized.nii.gz'))
        if os.path.exists(output_path):
            print("Output file already exists for", filename)
            continue         # Skip this file and move on to the next one
        
        print("Loading data for file:", filename)  # add this line
      
        # Load the image
        img_path = os.path.join(img_dir_pet, filename)
        img = nib.load(img_path)

        # Get the image data and header
        img_data = img.get_fdata()
        img_header = img.header.copy()

        #print("old_direction:",img.GetDirection())
        #print("old_origin:" ,img.GetOrigin())
        #print("old_spacing:" ,img.GetSpacing())
        #print("old_size:" ,img.GetSize())
        # Normalize the image data
        img_data_norm = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

        # Create a new image with the normalized data and the original header
        img_norm = nib.Nifti1Image(img_data_norm, img.affine, header=img_header)

        #print("new_direction:",img_norm.GetDirection())
        #print("new_origin:" ,img_norm.GetOrigin())
        #print("new_spacing:" ,img_norm.GetSpacing())
        #print("new_size:" ,img_norm.GetSize())

        # Save the normalized image to the Resampled_DWI directory
        img_norm_path = os.path.join("/mnt/nfs-students/PET_resampled_normalized/fixedmedianheader2/", filename.replace(".nii.gz", "_normalized.nii.gz"))
        #os.makedirs(os.path.dirname(img_norm_path), exist_ok=True)
        nib.save(img_norm, img_norm_path)
        print("PET normalized of patient:",filename)
        # Release the file handle
        del img_norm
        img.uncache()
'''
# Loop over all files in the directory Label images
for filename in os.listdir(img_dir_labels):
    if filename == "N38_SUV_resampled.nii.gz":
        print("Skipping file:", filename)
        continue
    print("next round")

    if filename.endswith(".nii.gz"):
        # Check if the output file already exists
        output_folder = '/mnt/nfs-students/PET_resampled_normalized/onLAVAN10/'
        output_path = os.path.join(output_folder, filename.replace('_resampled.nii.gz', '_resampled_normalized.nii.gz'))
        if os.path.exists(output_path):
            print("Output file already exists for", filename)
            continue  # Skip this file and move on to the next one
        # Load the image
        img_path = os.path.join(img_dir_labels, filename)
        img = nib.load(img_path)

        # Get the image data and header
        img_data = img.get_fdata()
        img_header = img.header.copy()

        # Normalize the image data
        img_data_norm = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

        # Create a new image with the normalized data and the original header
        img_norm = nib.Nifti1Image(img_data_norm, img.affine, header=img_header)

        # Save the normalized image to the Resampled_DWI directory
        img_norm_path = os.path.join("/mnt/nfs-students/Labels_resampled_normalized/onLAVAN10/", filename.replace(".nii.gz", "_normalized.nii.gz"))
        #os.makedirs(os.path.dirname(img_norm_path), exist_ok=True)
        nib.save(img_norm, img_norm_path)
        print("Label normalized of patient:",filename)

'''
'''
# Loop over all files in the directory Labels images
for filename in os.listdir(img_dir_labels):
    print("next round")
    if filename.endswith(".nii.gz"):
        # Load the image
        print("load Label image",filename)

        img_path = os.path.join(img_dir_labels, filename)
        img = nib.load(img_path)

        # Get the image data and header
        img_data = img.get_fdata()
        img_header = img.header.copy()
        print("Normalize the image data")
        # Normalize the image data
        img_data_norm = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
        print("Create a new image")
        # Create a new image with the normalized data and the original header
        img_norm = nib.Nifti1Image(img_data_norm, img.affine, header=img_header)
        print("save the new image")
        # Save the normalized image to the Resampled_DWI directory
        output_folder = '/mnt/nfs-students/schoensophie/Labels_resampled_normalized/onLAVA_N10/'
        output_path = os.path.join(output_folder, filename.replace('.nii.gz', '_normalized.nii.gz'))
        #print("image norm path=",img_norm_path)
        #sitk.WriteImage(img_norm, output_path)

        #os.makedirs(os.path.dirname(img_norm_path), exist_ok=True)
        nib.save(img_norm, output_path)
        print("Label normalized of patient:",filename)
# error files are red and not visible in the output path 
'''
print("end normalization")             

#schoensophie/nnUNet_raw/Dataset005_PETDWIfixedmedianheader/labelsTr