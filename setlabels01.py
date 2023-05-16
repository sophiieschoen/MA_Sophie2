'''
import pip
#pip install nibabel
#pip install nilearn
import numpy as np
import nibabel as nib


# Load the NIfTI image data
originalLabel = nib.load('/scratch/users/sschoen/original_Labels_nii/N10.nii.gz') #=DWI

data = originalLabel.get_fdata()

# Threshold the image
data[data > 0] = 1

# Save the binary image
binary_img = nib.Nifti1Image(data, originalLabel.affine, originalLabel.header)
nib.save(binary_img, 'original_Labels_binary/Labels_binary_N10.nii.gz')
'''

# as loop through whole folder:
import os
import numpy as np
import nibabel as nib

# Set the input and output directories
input_dir = '/mnt/nfs-students/Labels_resampled/fixedmedianheader/'
output_dir = '/mnt/nfs-students/Labels_resampled_binary/fixedmedianheader/'

print("Output directory:", output_dir)

# Loop through all the files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.nii.gz'):
        print("Check if the output file already exists of patient",filename)
        #output_folder = '/mnt/nfs-students/Labels_resampled/onLAVAN10/'
        output_path = os.path.join(output_dir, filename.replace('.nii.gz', '_binary.nii.gz'))
        print("Output path:", output_path)

        if os.path.exists(output_path):
            print("Output file already exists for", filename)
            continue  # Skip this file and move on to the next one

        print("Loading data for file:", filename)  # add this line

        # Load the NIfTI image data
        originalLabel = nib.load(os.path.join(input_dir, filename))

        # Threshold the image
        data = originalLabel.get_fdata()
        data[data > 0] = 1

        # Save the binary image with the same name as the input data
        binary_filename = os.path.join(output_dir, filename.replace('.nii.gz', '_binary.nii.gz'))
        binary_img = nib.Nifti1Image(data, originalLabel.affine, originalLabel.header)
        nib.save(binary_img, binary_filename)
        print("Labels set to binary of patient:",filename)

print("All Labels are binarized")
