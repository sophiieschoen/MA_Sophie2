import os
import SimpleITK as sitk
import numpy as np

# Define the paths to the two folders containing the images
folder1 = '/mnt/nfs-students/PET_nii'
folder2 = '/mnt/nfs-students/DWI_nii'

def find_maximum_size(folder1, folder2):
    max_sizes = np.zeros((3,))
    for folder in [folder1, folder2]:
        for filename in os.listdir(folder):
            print("look at this patient:",filename)
            try:
                image = sitk.ReadImage(os.path.join(folder, filename))
                sizes = np.array(image.GetSize())
                max_sizes = np.maximum(max_sizes, sizes)
            except:
                print(f"Error reading {filename}. Skipping...")
    return max_sizes

max_sizes = find_maximum_size(folder1, folder2)

# Print the maximum size in each dimension
print('Maximum size: ', max_sizes)
