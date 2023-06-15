import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt

def calculate_dice(true_path, pred_path):
    # Get list of all files in each directory
    true_files = os.listdir(true_path)
    pred_files = os.listdir(pred_path)
    
    dice_scores = []
    file_names = []

    for true_file, pred_file in zip(sorted(true_files), sorted(pred_files)):
        # Ignore hidden or system files
        if true_file.startswith('.') or pred_file.startswith('.'):
            continue

        # Load ground truth and predictions
        true_label = nib.load(os.path.join(true_path, true_file)).get_fdata()
        pred_label = nib.load(os.path.join(pred_path, pred_file)).get_fdata()

        # Flatten arrays
        true_label = true_label.flatten()
        pred_label = pred_label.flatten()

        # Calculate dice score
        intersection = np.sum(true_label * pred_label)
        dice_score = (2. * intersection) / (np.sum(true_label) + np.sum(pred_label))
        
        print(f"Dice score for {true_file}: {dice_score}")
        
        dice_scores.append(dice_score)
        file_names.append(true_file)
        
    return file_names, dice_scores

## ID7 before postprocessing ##
# Specify the directories
true_path = '/mnt/students/nnUNet_results_analzation/ID3_predictedlabels/groundtruth_labels_testset/'
#pred_path = '/mnt/students/nnUNet_results_analzation/ID3_predictedlabels/predicted_labels/'
pred_path = '/mnt/students/nnUNet_results_analzation/ID7_predictedlabels/predicted_labels/'

ID7_file_names, ID7_dice_scores = calculate_dice(true_path, pred_path)

# Print average dice score
print("Average Dice Score PET validationset preprocessed: ", np.mean(ID7_dice_scores))

#
#
#
## PET ID7 postprocessed ###
# Specify the directories
true_path = '/mnt/students/nnUNet_results_analzation/ID3_predictedlabels/groundtruth_labels_testset/'
#pred_path = '/mnt/students/nnUNet_results_analzation/ID2_predictedlabels/predicted_labels2/'
pred_path = '/mnt/students/nnUNet_results_analzation/ID7_predictedlabels/predicted_labels_postprocessed/'

petpost_file_names, petpost_dice_scores = calculate_dice(true_path, pred_path)

# Print average dice score
print("Average Dice Score PET testset: ", np.mean(petpost_dice_scores))

#

# Create a scatter plot
plt.figure(figsize=(8, 4))
plt.scatter(ID7_file_names, ID7_dice_scores, color='red', label='PET before')
plt.scatter(petpost_file_names, petpost_dice_scores, color='green', label='PET postprocessed')
#plt.scatter(petdwi_file_names, petdwi_dice_scores, color='blue', label='PET-DWI')
#plt.scatter(decfus_file_names, decfus_dice_scores, color='orange', label='decisionfusion')


plt.xticks(rotation=90)
plt.ylabel('Dice Score')
plt.title('Dice Score for PET validationset before vs. after postprocessing')
plt.legend()
 
# Save the scatter plot
#plt.savefig('/path/to/save/scatter_plot.png')

# Show the scatter plot
plt.show()

plt.savefig('/mnt/students/nnUNet_results_analzation/scatter_plot_postprocessedbeforevsafter.png')
