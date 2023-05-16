import os
import SimpleITK as sitk
#import nibabel as nib

##with nibabel isntead of sitk
'''
# Load the image data
dwi_image = nib.load(dwi_image_path)
# Get the image data as a numpy array
dwi_data = dwi_image.get_fdata()
# Get the image header information
dwi_header = dwi_image.header
'''

print("Resampling starting")


# Specify the input and output folders
#dwi_folder = '/users/sophie/thesis/datasets/smb_workfiles/schoensophie/DWI_nii/'
#pet_folder = '/users/sophie/thesis/datasets/smb_workfiles/schoensophie/PET_nii/'
#label_folder = '/users/sophie/thesis/datasets/smb_workfiles/schoensophie/Labels_nii/'
#lava_image_path = '/Users/Sophie/thesis/localdata/N10_LAVA.nii.gz'

dwi_folder = '/mnt/nfs-students/DWI_nii/'
pet_folder = '/mnt/nfs-students/PET_nii/'
label_folder = '/mnt/nfs-students/Labels_nii/'
#lava_image_path = '/mnt/nfs-students/LAVA_nii/N10_LAVA.nii.gz'

#dwi_image = sitk.ReadImage('/mnt/nfs-students/DWI_nii/N12-DWI-800.nii.gz')
#pet_image = '/mnt/nfs-students/PET_nii/N12_SUV.nii.gz'

print("specified folders")

'''
old_size=dwi_image.GetSize()
old_spacing=dwi_image.GetSpacing()


print("Original size of dwi image",dwi_image.GetSize())
print("Original spacing of dwi image",dwi_image.GetSpacing())
print("Original direction of dwi image",dwi_image.GetDirection())
print("Original origin of dwi image",dwi_image.GetOrigin())

#spacing = dwi_image.GetSpacing()
direction = dwi_image.GetDirection()
origin = dwi_image.GetOrigin()


#size = lava_image.GetSize()
#spacing = lava_image.GetSpacing()
#direction = lava_image.GetDirection()
#origin = lava_image.GetOrigin()


# Load the LAVA image as the reference image
#lava_image = sitk.ReadImage(lava_image_path)

# Get the desired size, spacing, direction, and origin from the LAVA image
#size = lava_image.GetSize()
#spacing = lava_image.GetSpacing()

'''
new_spacing=(1.8, 1.8, 2.8)
max_size=(512,512,674)
direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
origin=(-298.4375,-278.4375, -1577.86999512)

#new_size = [int(old_size[i] * old_spacing[i] / new_spacing[i]) for i in range(3)]


#direction = lava_image.GetDirection()
#origin = lava_image.GetOrigin()

print("get header done")

#print("size:",size)
#print("set new spacing to:",new_spacing)
#print("direction",direction)
#print("origin",origin)

'''

# Create a resampling filter
resampler = sitk.ResampleImageFilter()

# Set the resampling parameters
resampler.SetSize(max_size)
resampler.SetOutputSpacing(new_spacing)
resampler.SetOutputDirection(direction)
resampler.SetOutputOrigin(origin)
#resampler.SetTransform(sitk.Transform())
print("Output Spacing:", resampler.GetOutputSpacing())
print("Output Direction:", resampler.GetOutputDirection())
print("Output Origin:", resampler.GetOutputOrigin())
print("Output Size:", resampler.GetSize())
'''
#print("Transform:", resampler.GetTransform())
'''
print("resample the dwi on the new spacing")
dwi_resampled = resampler.Execute(dwi_image)

print("New size of dwi image",dwi_resampled.GetSize())
print("New spacing of dwi image",dwi_resampled.GetSpacing())
print("New direction of dwi image",dwi_resampled.GetDirection())
print("New origin of dwi image",dwi_resampled.GetOrigin())

print("save resampled image")
output_path = '/mnt/nfs-students/DWI_resampled/onmedianspacingandsize/DWI_N12_resampled_medianspacing_maxsize.nii.gz'
sitk.WriteImage(dwi_resampled, output_path)
'''
#print("Resampling parameters set")


# Loop over all DWI images in the input folder and resample them
# Loop over all DWI images in the input folder and resample them

for filename in os.listdir(dwi_folder):
    if filename.endswith('.nii.gz'):
      # Check if the output file already exists
        output_folder = '/mnt/nfs-students/DWI_resampled/fixedmedianheader2/'
        output_path = os.path.join(output_folder, filename.replace('.nii.gz', '_resampled.nii.gz'))
        if os.path.exists(output_path):
            print("Output file already exists for", filename)
            continue  # Skip this file and move on to the next one  
       
        dwi_image_path = os.path.join(dwi_folder, filename)
        dwi_image = sitk.ReadImage(dwi_image_path)
        print("old_direction:",dwi_image.GetDirection())
        print("old_origin:" ,dwi_image.GetOrigin())
        #dwi_image = nib.load(dwi_image_path)
         # Create a resampling filter
        resampler = sitk.ResampleImageFilter()
        # Set the resampling parameters
        #direction = dwi_image.GetDirection()
        #origin = dwi_image.GetOrigin()
        resampler.SetSize(max_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputDirection(direction)
        resampler.SetOutputOrigin(origin)
        dwi_resampled = resampler.Execute(dwi_image)
        print("new_direction:",dwi_resampled.GetDirection())
        print("new_origin:" ,dwi_resampled.GetOrigin())
        sitk.WriteImage(dwi_resampled, output_path)
        print("DWI resampled of patient:",filename)

print("Resampling DWI completed")

# Loop over all PET images in the input folder and resample them
for filename in os.listdir(pet_folder):
    if filename.endswith('.nii.gz'):
         # Check if the output file already exists
        output_folder = '/mnt/nfs-students/PET_resampled/fixedmedianheader2/'
        output_path = os.path.join(output_folder, filename.replace('.nii.gz', '_resampled.nii.gz'))
        if os.path.exists(output_path):
            print("Output file already exists for PET", filename)
            continue  # Skip this file and move on to the next one

        pet_image_path = os.path.join(pet_folder, filename)
        pet_image = sitk.ReadImage(pet_image_path)
        resampler = sitk.ResampleImageFilter()
        # Set the resampling parameters
        #direction = pet_image.GetDirection()
        #origin = pet_image.GetOrigin()
        print("old_direction:",pet_image.GetDirection())
        print("old_origin:" ,pet_image.GetOrigin())
        resampler.SetSize(max_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputDirection(direction)
        resampler.SetOutputOrigin(origin)
        pet_resampled = resampler.Execute(pet_image)
        #print("outputfolder:",output_folder)
        #print("outputpath:",output_path)
        print("new_direction:",pet_resampled.GetDirection())
        print("new_origin:" ,pet_resampled.GetOrigin())
        #output_folder = '/mnt/nfs-students/PET_resampled/onmedianspacingandmaxsize/'
        #output_path = os.path.join(output_folder, filename.replace('.nii.gz', '_resampled.nii.gz'))
        sitk.WriteImage(pet_resampled, output_path)       
        print("PET resampled of patient:",filename)


print("Resampling PET completed")


# Loop over all label images in the input folder and resample them
for filename in os.listdir(label_folder):
    if filename.endswith('.nii.gz'):
        # Check if the output file already exists
        output_folder = '/mnt/nfs-students/Labels_resampled/fixedmedianheader2/'
        output_path = os.path.join(output_folder, filename.replace('.nii.gz', '_resampled.nii.gz'))
        if os.path.exists(output_path):
            print("Output file already exists for Label", filename)
            continue  # Skip this file and move on to the next one

        label_image_path = os.path.join(label_folder, filename)
        label_image = sitk.ReadImage(label_image_path)
        resampler = sitk.ResampleImageFilter()

        # Set the resampling parameters
        #direction = label_image.GetDirection()
        #origin = label_image.GetOrigin()
        print("old_direction:",label_image.GetDirection())
        print("old_origin:" ,label_image.GetOrigin())
        resampler.SetSize(max_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputDirection(direction)
        resampler.SetOutputOrigin(origin)
        label_resampled = resampler.Execute(label_image)
        print("new_direction:",label_resampled.GetDirection())
        print("new_origin:" ,label_resampled.GetOrigin())
        sitk.WriteImage(label_resampled, output_path)
        #print("outputpath:",output_path)
        print("Label resampled of patient:",filename)


print("Resampling Label completed")

print("Resampling completed")
