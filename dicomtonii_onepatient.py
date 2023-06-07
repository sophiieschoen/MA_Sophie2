

import dicom2nifti
import os
from glob import glob


# convert one patient:
path_one_patient= '/mnt/students/LAVA_dicom/N29-LAVA'
path_out_data='/mnt/students/LAVA_nii'

#dicom2nifti.dicom_series_to_nifti(path_one_patient,os.path.join(path_out_data,'N29_test3.nii.gz'))

import SimpleITK as sitk

# Load DICOM series
reader = sitk.ImageSeriesReader()
dicom_files = reader.GetGDCMSeriesFileNames(path_one_patient)
reader.SetFileNames(dicom_files)
dicom_series = reader.Execute()

# Resample to have uniform spacing
resampler = sitk.ResampleImageFilter()
new_spacing = (1.8, 1.8, 2.8)  # Set your desired spacing
resampler.SetOutputSpacing(new_spacing)
resampler.SetSize((512, 512, 674))  # Set your desired size
resampled_series = resampler.Execute(dicom_series)

# Convert to Nifti
sitk.WriteImage(resampled_series, os.path.join(path_out_data, 'N29_test3.nii.gz'))
print("saved nifti file")

##all patients:
#path_alldicom='/Users/Sophie/MA_CancerImaging/NewDataset_LAVA_dicom'
#path_out_allnifti='/Users/Sophie/MA_CancerImaging/NewDataset_LAVA_nii'
#
#dicom2nifti.convert_directory(path_alldicom, path_out_allnifti)
#
## convert more patients at the same time
#for i, patient in enumerate(glob(path_alldicom)):
#    dicom2nifti.dicom_series_to_nifti(patient,os.path.join(path_out_allnifti,'patient_test_'+str(i+1)+'.nii.gz'))






















#import os
#import pydicom
#import numpy as np
#import nibabel as nib
#
#dicom_directory = '/mnt/students/LAVA_dicom/N29-LAVA'
#output_directory = '/mnt/students/LAVA_nii'
#
## List all DICOM files in the patient directory
#dicom_files = [filename for filename in os.listdir(dicom_directory) if filename.endswith('.dcm')]
#
#if len(dicom_files) == 0:
#    print("No DICOM files found in", dicom_directory)
#else:
#    dicom_slices = []
#
#    for filename in dicom_files:
#        file_path = os.path.join(dicom_directory, filename)
#        try:
#            dcm = pydicom.dcmread(file_path)
#            dicom_slices.append(dcm.pixel_array)
#            print("Successfully loaded DICOM file:", filename)
#        except:
#            print("Failed to load DICOM file:", filename)
#
#    if len(dicom_slices) == 0:
#        print("No valid DICOM slices found in", dicom_directory)
#    else:
#        # Create a 3D volume from the DICOM slices
#        dicom_volume = np.stack(dicom_slices, axis=-1)
#
#        # Convert the DICOM volume to NIfTI format
#        nifti_img = nib.Nifti1Image(dicom_volume, affine=np.eye(4))
#
#        # Save the NIfTI image
#        output_filename = "N29-LAVA.nii.gz"
#        output_path = os.path.join(output_directory, output_filename)
#        nib.save(nifti_img, output_path)
#
#        print("NIfTI image saved:", output_filename)
#
#
#


##### Versuch 2:



#import os
#import glob
#import pydicom
#import nibabel as nib
#import numpy as np
#
#def convert_dicom_to_nifti(dicom_folder, output_path):
#    # Get a list of all DICOM files in the folder
#    dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
#
#    # Load the first DICOM file to get the necessary information for NIfTI
#    first_dicom = pydicom.read_file(dicom_files[0])
#    image_data = first_dicom.pixel_array
#    shape = image_data.shape
#
#    # Create an empty 3D array to store the pixel data
#    data = np.zeros(shape + (len(dicom_files),))
#
#    # Loop over all DICOM files and fill the 3D array
#    for i, dicom_file in enumerate(dicom_files):
#        dicom = pydicom.read_file(dicom_file)
#        data[..., i] = dicom.pixel_array
#
#    # Get the affine transformation matrix from DICOM metadata
#    affine = np.diag([float(first_dicom.PixelSpacing[0]),
#                      float(first_dicom.PixelSpacing[1]),
#                      float(first_dicom.SliceThickness),
#                      1.0])
#    affine[:3, -1] = [-x for x in first_dicom.ImagePositionPatient]
#
#    # Create a NIfTI image from the pixel data
#    nifti_image = nib.Nifti1Image(data, affine)
#    
#    # Save the NIfTI image to the specified output path
#    nib.save(nifti_image, output_path)
#
#    # Print the number of DICOM files used
#    print("Number of DICOM files used:", len(dicom_files))
#    print("saved nifti file of patient:",nifti_image)
## Example usage
#dicom_folder = "/mnt/students/LAVA_dicom/N29-LAVA"
#output_path = "/mnt/students/LAVA_nii/N29-LAVA_test_compressed.nii.gz"
#
#convert_dicom_to_nifti(dicom_folder, output_path)
#