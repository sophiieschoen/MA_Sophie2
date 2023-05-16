print("start resampling")

import SimpleITK as sitk
print(sitk.Version())

# Load the images
#dwi_image = sitk.ReadImage('/users/sophie/thesis/datasets/smb_workfiles/schoensophie/DWI_nii/N12-DWI-800.nii.gz')
dwi_image = sitk.ReadImage('/mnt/nfs-students/DWI_nii/N12-DWI-800.nii.gz')


#pet_image = sitk.ReadImage('/users/sophie/thesis/datasets/smb_workfiles/schoensophie/PET_nii/N12_SUV.nii.gz')
#label_image = sitk.ReadImage('/users/sophie/thesis/datasets/smb_workfiles/schoensophie/Labels_nii/N12.nii.gz')
#lava_image = sitk.ReadImage('/Users/Sophie/thesis/localdata/N10_LAVA.nii.gz')
dwi_image = sitk.ReadImage('/mnt/nfs-students/DWI_nii/N12-DWI-800.nii.gz')
pet_image = sitk.ReadImage('/mnt/nfs-students/PET_nii/N12_SUV.nii.gz')
label_image = sitk.ReadImage('/mnt/nfs-students/Labels_nii/N12.nii.gz')
lava_image = sitk.ReadImage('/mnt/nfs-students/LAVA_nii/N10_LAVA.nii.gz')

print("got files")

# Get the desired size, spacing, direction, and origin from the LAVA image
size = lava_image.GetSize()
spacing = lava_image.GetSpacing()
direction = lava_image.GetDirection()
origin = lava_image.GetOrigin()

# Create a resampling filter
resampler = sitk.ResampleImageFilter()

# Set the resampling parameters
resampler.SetSize(size)
resampler.SetOutputSpacing(spacing)
resampler.SetOutputDirection(direction)
resampler.SetOutputOrigin(origin)
resampler.SetTransform(sitk.Transform())


print("start resampling")
# Resample the DWI image
dwi_resampled = resampler.Execute(dwi_image)
print("dwi resampled")
# Resample the PET image
pet_resampled = resampler.Execute(pet_image)
print("pet resampled")

# Resample the label image
label_resampled = resampler.Execute(label_image)
print("label resampled")

# Save the resampled images in the specified folders
#sitk.WriteImage(dwi_resampled, '/users/sophie/thesis/datasets/smb_workfiles/schoensophie/DWI_resampled/onLAVAN10/N12dwi_resampled.nii.gz')
sitk.WriteImage(dwi_resampled, '/mnt/nfs-students/DWI_resampled/onLAVAN10/N12dwi_resampled.nii.gz')
print("saved dwi")

#sitk.WriteImage(pet_resampled, '/users/sophie/thesis/datasets/smb_workfiles/schoensophie/PET_resampled/onLAVAN10/N12pet_resampled.nii.gz')
sitk.WriteImage(pet_resampled, '/mnt/nfs-students/PET_resampled/onLAVAN10/N12pet_resampled.nii.gz')
sitk.WriteImage(label_resampled, '/mnt/nfs-students/Labels_resampled/onLAVAN10/N12label_resampled.nii.gz')

print("resampling completed")
