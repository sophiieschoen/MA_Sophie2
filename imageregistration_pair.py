#registers the DWI image to the PET image to the label imgage of a patient but not all patients onto each other
import SimpleITK as sitk
import os

def register_images(fixed_image_path, moving_image_path, output_dir):
    # Load the images
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # Create the registration object
    registration = sitk.ImageRegistrationMethod()

    # Set the initial transform
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration.SetInitialTransform(initial_transform)

    # Set the similarity metric
    registration.SetMetricAsMattesMutualInformation()

    # Set the optimizer
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )

    # Set the interpolator
    registration.SetInterpolator(sitk.sitkLinear)

    # Set the multi-resolution strategy
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])

    # Perform the registration
    transform = registration.Execute(fixed_image, moving_image)

    # Resample the moving image onto the fixed image grid
    resampled_image = sitk.Resample(
        moving_image, fixed_image, transform,
        sitk.sitkLinear, 0.0, moving_image.GetPixelID()
    )

    # Save the registered image
    filename = os.path.basename(moving_image_path)
    output_path = os.path.join(output_dir, "registered_" + filename)
    sitk.WriteImage(resampled_image, output_path)

    # Load the fixed image to use as reference
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)

    # Load the registered image
    registered_image = sitk.ReadImage(output_path, sitk.sitkFloat32)

    # Set the desired image attributes based on the fixed image
    registered_image.SetSpacing(fixed_image.GetSpacing())
    registered_image.SetOrigin(fixed_image.GetOrigin())
    registered_image.SetDirection(fixed_image.GetDirection())

    # Save the updated registered image
    sitk.WriteImage(registered_image, output_path)

    return output_path

# Set the paths to your DWI, PET, and label images
dwi_dir = "/path/to/dwi/images/"
pet_dir = "/path/to/pet/images/"
label_dir = "/path/to/label/images/"

# Set the output directory for the registered images
output_dir = "/path/to/output/directory/"

# Get the list of DWI, PET, and label image files
dwi_files = os.listdir(dwi_dir)
pet_files = os.listdir(pet_dir)
label_files = os.listdir(label_dir)

# Register each DWI, PET, and label image pair
for dwi_file in dwi_files:
    dwi_path = os.path.join(dwi_dir, dwi_file)

    for pet_file in pet_files:
        pet_path = os.path.join(pet_dir, pet_file)

        for label_file in label_files:
            label_path = os.path.join(label_dir, label_file)

            # Perform registration
            registered_image_path = register_images(dwi_path, pet_path, output_dir)
