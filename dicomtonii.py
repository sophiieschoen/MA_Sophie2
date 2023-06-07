#import os
#import pydicom
#import nibabel as nib
#import numpy as np
#
#directory = '/mnt/students/LAVA_dicom'
#output_directory = '/mnt/students/LAVA_nii'
#
## List all patient directories in the main directory
#patient_directories = [dir for dir in os.listdir(directory) if os.path.isdir(os.path.join(directory, dir))]
#
#if len(patient_directories) == 0:
#    print("No patient directories found in", directory)
#else:
#    for patient_dir in patient_directories:
#        patient_dir_path = os.path.join(directory, patient_dir)
#
#        # List all DICOM files in the patient directory
#        dicom_files = [filename for filename in os.listdir(patient_dir_path) if filename.endswith('.dcm')]
#
#        if len(dicom_files) == 0:
#            print("No DICOM files found in", patient_dir_path)
#            continue  # Skip to the next patient directory if no DICOM files are found
#
#        dicom_slices = []
#
#        for filename in dicom_files:
#            file_path = os.path.join(patient_dir_path, filename)
#            try:
#                dcm = pydicom.dcmread(file_path)
#                dicom_slices.append(dcm.pixel_array)
#                print("Successfully loaded DICOM file:", filename)
#            except:
#                print("Failed to load DICOM file:", filename)
#
#        if len(dicom_slices) == 0:
#            print("No valid DICOM slices found in", patient_dir_path)
#            continue  # Skip to the next patient directory if no valid DICOM slices are found
#
#        # Create a 3D volume from the DICOM slices
#        dicom_volume = np.stack(dicom_slices, axis=-1)
#
#        # Convert the DICOM volume to NIfTI format
#        nifti_img = nib.Nifti1Image(dicom_volume, affine=np.eye(4))
#
#        # Save the NIfTI image
#        output_filename = patient_dir + '.nii.gz'
#        output_path = os.path.join(output_directory, output_filename)
#        nib.save(nifti_img, output_path)
#
#        print("NIfTI image saved:", output_filename)
#

##this code does not detect if the file ending is ".dcm": 
import os
import pydicom
import nibabel as nib
import numpy as np

directory = '/mnt/students/LAVA_dicom'
output_directory = '/mnt/students/LAVA_nii'

# List all patient directories in the main directory
patient_directories = [dir for dir in os.listdir(directory) if os.path.isdir(os.path.join(directory, dir))]

if len(patient_directories) == 0:
    print("No patient directories found in", directory)
else:
    for patient_dir in patient_directories:
        patient_dir_path = os.path.join(directory, patient_dir)

        # List all files in the patient directory
        files = [filename for filename in os.listdir(patient_dir_path)]

        if len(files) == 0:
            print("No files found in", patient_dir_path)
            continue  # Skip to the next patient directory if no files are found

        dicom_slices = []

        for filename in files:
            file_path = os.path.join(patient_dir_path, filename)
            try:
                dcm = pydicom.dcmread(file_path)

                # Check if the file is a valid DICOM file
                if dcm.pixel_array is not None:
                    dicom_slices.append(dcm.pixel_array)
                    print("Successfully loaded DICOM file:", filename)
            except pydicom.errors.InvalidDicomError:
                print("Not a valid DICOM file:", filename)

        if len(dicom_slices) == 0:
            print("No valid DICOM slices found in", patient_dir_path)
            continue  # Skip to the next patient directory if no valid DICOM slices are found

        # Create a 3D volume from the DICOM slices
        dicom_volume = np.stack(dicom_slices, axis=-1)

        # Convert the DICOM volume to NIfTI format
        nifti_img = nib.Nifti1Image(dicom_volume, affine=np.eye(4))

        # Save the NIfTI image
        output_filename = patient_dir + '.nii.gz'
        output_path = os.path.join(output_directory, output_filename)
        nib.save(nifti_img, output_path)

        print("NIfTI image saved:", output_filename)
