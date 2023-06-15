

import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt


def dice_coefficient(seg_mask, ground_truth):
    intersection = np.sum(seg_mask * ground_truth)
    total = np.sum(seg_mask) + np.sum(ground_truth)
    dice_score = (2.0 * intersection) / (total + 1e-7)  # Adding a small epsilon to avoid division by zero
    return dice_score

def calculate_dice_scores(true_path, pred_path):
    true_files = os.listdir(true_path)
    pred_files = os.listdir(pred_path)

    dice_scores = []
    file_names = []

    for true_file in sorted(true_files):
        if true_file.startswith('.'):
            continue

        true_nifti = nib.load(os.path.join(true_path, true_file))
        true_label = true_nifti.get_fdata()

        pred_file = true_file[:-7] + '.nii.gz'
        if pred_file in pred_files:
            pred_nifti = nib.load(os.path.join(pred_path, pred_file))
            pred_label = pred_nifti.get_fdata()

            intersection = np.sum(true_label * pred_label)
            dice_score = (2.0 * intersection) / (np.sum(true_label) + np.sum(pred_label))

            print(f"Dice score for {true_file}: {dice_score}")

            dice_scores.append(dice_score)
            file_names.append(true_file)

    return file_names, dice_scores

# DWI
true_path = '/mnt/students/nnUNet_results_analzation/ID3_predictedlabels/groundtruth_labels_testset/'
pred_path = '/mnt/students/nnUNet_results_analzation/ID6_predictedlabels/predicted_labels/'

dwi_file_names, dwi_dice_scores = calculate_dice_scores(true_path, pred_path)

print("Average Dice Score DWI testset:", np.mean(dwi_dice_scores))

# PET
true_path = '/mnt/students/nnUNet_results_analzation/ID3_predictedlabels/groundtruth_labels_testset/'
pred_path = '/mnt/students/nnUNet_results_analzation/ID7_predictedlabels/predicted_labels/'

pet_file_names, pet_dice_scores = calculate_dice_scores(true_path, pred_path)

print("Average Dice Score PET testset:", np.mean(pet_dice_scores))

# PET-DWI
true_path = '/mnt/students/nnUNet_results_analzation/ID3_predictedlabels/groundtruth_labels_testset/'
pred_path = '/mnt/students/nnUNet_results_analzation/ID5_predictedlabels/predicted_labels/'

petdwi_file_names, petdwi_dice_scores = calculate_dice_scores(true_path, pred_path)

print("Average Dice Score PETDWI testset:", np.mean(petdwi_dice_scores))

#### average decisionfusion #### 
# Specify the directories
true_path = '/mnt/students/nnUNet_results_analzation/ID3_predictedlabels/groundtruth_labels_testset/'
#pred_path = '/mnt/students/nnUNet_results_analzation/ID14_predictedlabels/predicted_labels/'
pred_path = '/mnt/students/nnUNet_results_analzation/decisionlevelfusionID6und7/averagefusion/'


decfus_file_names, decfus_dice_scores = calculate_dice_scores(true_path, pred_path)

# Print average dice score
print("Average Dice Score decision fusion testset: ", np.mean(decfus_dice_scores))

# Create a scatter plot
plt.figure(figsize=(8, 4))
plt.scatter(dwi_file_names, dwi_dice_scores, color='red', label='DWI')
plt.scatter(pet_file_names, pet_dice_scores, color='green', label='PET')
plt.scatter(petdwi_file_names, petdwi_dice_scores, color='blue', label='PET-DWI')
plt.scatter(decfus_file_names, decfus_dice_scores, color='orange', label='decisionfusion')


plt.xticks(rotation=90)
plt.ylabel('Dice Score')
plt.title('Dice Score for PET, DWI, PET-DWI and decisionfusion')
plt.legend()
 
# Save the scatter plot
#plt.savefig('/path/to/save/scatter_plot.png')

# Show the scatter plot
plt.show()

plt.savefig('/mnt/students/nnUNet_results_analzation/scatter_plot_testsetcomparison_withdecisionfusion_test2.png')


##for single image
#import numpy as np
#import nibabel as nib
#
#def dice_coefficient(seg_mask, ground_truth):
#    intersection = np.sum(seg_mask * ground_truth)
#    total = np.sum(seg_mask) + np.sum(ground_truth)
#    dice_score = (2.0 * intersection) / (total + 1e-7)  # Adding a small epsilon to avoid division by zero
#    return dice_score
#
## Load predicted segmentation and ground truth label from NIfTI files
#predicted_nifti = nib.load('/mnt/students/nnUNet_results_analzation/ID3_predictedlabels/groundtruth_labels_testset/la_088.nii.gz')
#ground_truth_nifti = nib.load('/mnt/students/nnUNet_results_analzation/ID5_predictedlabels/predicted_labels/la_088.nii.gz')
#
## Get data arrays from NIfTI images
#predicted_data = predicted_nifti.get_fdata()
#ground_truth_data = ground_truth_nifti.get_fdata()
#
## Ensure the data arrays have the same shape
#assert predicted_data.shape == ground_truth_data.shape, "Segmentation and ground truth shape mismatch"
#
## Calculate Dice score
#score = dice_coefficient(predicted_data, ground_truth_data)
#print("Dice score:", score)
#
######

