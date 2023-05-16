import os
import SimpleITK as sitk

# Directory containing the input images
img_dir_dwi = "/mnt/nfs-students/DWI_resampled/fixedmedianheader2/"
img_dir_pet = "/mnt/nfs-students/PET_resampled/fixedmedianheader2/"

# Output directories for normalized images
output_dir_dwi = "/mnt/nfs-students/DWI_resampled_normalized/fixedmedianheader2/"
output_dir_pet = "/mnt/nfs-students/PET_resampled_normalized/fixedmedianheader2/"

# Function for image normalization
def normalize_image(img):
    img_norm = sitk.Normalize(img)
    return img_norm

# Normalize DWI images
for filename in os.listdir(img_dir_dwi):
    if filename.endswith(".nii.gz"):
        # Check if the output file already exists
        output_path = os.path.join(output_dir_dwi, filename.replace('_resampled.nii.gz', '_resampled_normalized.nii.gz'))
        if os.path.exists(output_path):
            print("Output file already exists for", filename)
            continue  # Skip this file and move on to the next one

        # Load the image
        img_path = os.path.join(img_dir_dwi, filename)
        img = sitk.ReadImage(img_path)

        # Normalize the image
        img_norm = normalize_image(img)

        # Save the normalized image
        img_norm_path = os.path.join(output_dir_dwi, filename.replace(".nii.gz", "_normalized.nii.gz"))
        sitk.WriteImage(img_norm, img_norm_path)
        print("Normalized DWI image saved:", img_norm_path)

# Normalize PET images
for filename in os.listdir(img_dir_pet):
    if filename.endswith(".nii.gz"):
        # Check if the output file already exists
        output_path = os.path.join(output_dir_pet, filename.replace('_resampled.nii.gz', '_resampled_normalized.nii.gz'))
        if os.path.exists(output_path):
            print("Output file already exists for", filename)
            continue  # Skip this file and move on to the next one

        # Load the image
        img_path = os.path.join(img_dir_pet, filename)
        img = sitk.ReadImage(img_path)

        # Normalize the image
        img_norm = normalize_image(img)

        # Save the normalized image
        img_norm_path = os.path.join(output_dir_pet, filename.replace(".nii.gz", "_normalized.nii.gz"))
        sitk.WriteImage(img_norm, img_norm_path)
        print("Normalized PET image saved:", img_norm_path)
