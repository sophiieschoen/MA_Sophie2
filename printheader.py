import os
import SimpleITK as sitk

def print_image_info(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)
     # Sort the file list
    file_list.sort()

    # Iterate over each file
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        
        # Check if the file is an image (nii or nii.gz)
        if file_name.endswith(".nii") or file_name.endswith(".nii.gz"):
            try:
                # Load the image
                img = sitk.ReadImage(file_path)
                
                # Extract the direction matrix and origin
                direction_matrix = img.GetDirection()
                origin = img.GetOrigin()
                
                # Print the information
                print("File: ", file_name)
                print("Direction Matrix:\n", direction_matrix)
                print("Origin: ", origin)
                print("------------------------------------")
            except Exception as e:
                print("Error loading file:", file_name)
                print("Error message:", str(e))
                print("------------------------------------")

# Specify the folder path
folder_path = "/mnt/nfs-students/nnUNet_raw/Dataset007_PETfixedmedianheader/imagesTr/"
#folder_path = "/mnt/nfs-students/PET_resampled_normalized/fixedmedianheader2/"

# Call the function to print image information
print_image_info(folder_path)
