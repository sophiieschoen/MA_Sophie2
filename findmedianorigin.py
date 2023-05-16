import os
import SimpleITK as sitk
import numpy as np

# Define the paths to the two folders containing the images
folder2 = '/mnt/nfs-students/PET_nii'
#folder1 = '/mnt/nfs-students/DWI_nii'
folder3='/mnt/nfs-students/Labels_nii'


def find_median_origin(folder2,folder3):
    origins = []
    for folder in [folder2,folder3]:
        for filename in os.listdir(folder):
            print("look at this patient:",filename)
            try:
                image = sitk.ReadImage(os.path.join(folder, filename))
                origins.append(image.GetOrigin())
                print("Origin of this patient is:",image.GetOrigin())
            except:
                print(f"Error reading {filename}. Skipping...")
    median_origin = np.median(origins, axis=0)
    return median_origin


median_origin = find_median_origin(folder2,folder3)

# Print the median spacing
print('Median origin: ', median_origin)
