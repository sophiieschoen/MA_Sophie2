import dicom2nifti
import os
from glob import glob

path_out_data = '/mnt/students/LAVA_nii/LAVA_version2/'
path = '/mnt/students/LAVA_dicom/'

# Get a list of subdirectories (patient folders) in the specified path
patient_folders = glob(os.path.join(path, '*'))

for i, patient_folder in enumerate(patient_folders):
    # Generate the output NIfTI filename
    nifti_filename = 'patient_test_' + str(i + 1) + '.nii.gz'
    nifti_path = os.path.join(path_out_data, nifti_filename)

    # Convert DICOM series to NIfTI
    dicom2nifti.dicom_series_to_nifti(patient_folder, nifti_path)

    print(f"Converted: {patient_folder} -> {nifti_path}")
