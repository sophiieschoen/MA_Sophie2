import os
import SimpleITK as sitk
import numpy as np

# Define the paths to the two folders containing the images
folder1 = '/mnt/nfs-students/PET_nii'
folder2 = '/mnt/nfs-students/DWI_nii'


def find_median_size(folder1, folder2):
    sizes = []
    for folder in [folder1, folder2]:
        for filename in os.listdir(folder):
            print("look at this patient:",filename)
            try:
                image = sitk.ReadImage(os.path.join(folder, filename))
                sizes.append(image.GetSize())
            except:
                print(f"Error reading {filename}. Skipping...")
    median_sizes = np.median(sizes, axis=0)
    return median_sizes


median_sizes = find_median_size(folder1, folder2)

# Print the median spacing
print('Median size: ', median_sizes)
