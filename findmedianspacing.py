import os
import SimpleITK as sitk
import numpy as np

# Define the paths to the two folders containing the images
folder1 = '/mnt/nfs-students/PET_nii'
folder2 = '/mnt/nfs-students/DWI_nii'


def find_median_spacing(folder1, folder2):
    spacings = []
    for folder in [folder1, folder2]:
        for filename in os.listdir(folder):
            print("look at this patient:",filename)
            try:
                image = sitk.ReadImage(os.path.join(folder, filename))
                spacings.append(image.GetSpacing())
            except:
                print(f"Error reading {filename}. Skipping...")
    median_spacing = np.median(spacings, axis=0)
    return median_spacing


median_spacing = find_median_spacing(folder1, folder2)

# Print the median spacing
print('Median spacing: ', median_spacing)
