### PET #####

import json
import matplotlib.pyplot as plt

# Load the JSON file
with open('/mnt/students/nnUNet_results_analzation/ID7_predictedlabels/summary.json') as f:
    data = json.load(f)

# Extract patient IDs and Dice values
patient_ids_PET = []
dice_values_PET = []

for metrics in data['metric_per_case']:
    #print("data",data)
    prediction_file = metrics['prediction_file']
    patient_id = prediction_file.split('/')[-1].split('.')[0]
    dice_value = metrics['metrics']['1']['Dice']
    patient_ids_PET.append(patient_id)
    dice_values_PET.append(dice_value)
    
#print("patient IDs PET",patient_id_PET)  
#print("dice values PET",dice_value_PET)    
  

# Create scatter plot
plt.scatter(patient_ids_PET, dice_values_PET)
plt.xlabel('Patient ID')
plt.ylabel('Dice')
plt.title('PET - Dice Scores per Patient')
plt.xticks(rotation=45)
plt.show()

print("save scatterplot")

#Save the plot as an image file
##plt.savefig('/mnt/students/nnUNet_results_analzation/ID7_predictedlabels/scatter_plot.png')

#print median dice score:
import numpy as np

# Assuming you already have the dice_values array

median_dice_PET = np.median(dice_values_PET)

print("Median Dice Score PET:", median_dice_PET)

# Filter the array to exclude elements with a dice value of 0
filtered_dice_values = np.array([value for value in dice_values_PET if value != 0])

# Calculate the median of the filtered array
median_dice_PET = np.median(filtered_dice_values)
print("Median filtered Dice Score PET:", median_dice_PET)



## PET DWI

import json
import matplotlib.pyplot as plt

# Load the JSON file
with open('/mnt/students/nnUNet_results_analzation/ID5_predictedlabels/summary.json') as f:
    data = json.load(f)

# Extract patient IDs and Dice values
patient_ids_PETDWI = []
dice_values_PETDWI = []

for metrics in data['metric_per_case']:
    #print("data",data)
    prediction_file = metrics['prediction_file']
    patient_id = prediction_file.split('/')[-1].split('.')[0]
    dice_value = metrics['metrics']['1']['Dice']
    patient_ids_PETDWI.append(patient_id)
    dice_values_PETDWI.append(dice_value)
    
#print("patient IDs",patient_id_PETDWI)  
#print("dice values",dice_value_PETDWI)    
  

# Create scatter plot
plt.scatter(patient_ids_PETDWI, dice_values_PETDWI)
plt.xlabel('Patient ID')
plt.ylabel('Dice')
plt.title('PET - Dice Scores per Patient')
plt.xticks(rotation=45)
plt.show()

print("save scatterplot")

#Save the plot as an image file
#plt.savefig('/mnt/students/nnUNet_results_analzation/ID5_predictedlabels/scatter_plot.png')

#print median dice score:
import numpy as np

# Assuming you already have the dice_values array

median_dice_PETDWI = np.median(dice_values_PETDWI)

print("Median Dice Score PETDWI:", median_dice_PETDWI)

# Filter the array to exclude elements with a dice value of 0
filtered_dice_values = np.array([value for value in dice_values_PETDWI if value != 0])

# Calculate the median of the filtered array
median_dice_PETDWI = np.median(filtered_dice_values)
print("Median filtered Dice Score PETDWI:", median_dice_PETDWI)




## print both scatterplots in one:

# Create scatter plot
import matplotlib.pyplot as plt

plt.scatter(patient_ids_PETDWI, dice_values_PETDWI, color='green', label='PETDWI')
plt.scatter(patient_ids_PETDWI, dice_values_PET, color='orange', label='PET')
plt.xlabel('Patient ID')
plt.ylabel('Dice')
plt.title('PET vs. PETDWI - Dice Scores per Patient')
plt.xticks(rotation=45)
plt.legend()
plt.show()

#Save the plot as an image file
#plt.savefig('/mnt/students/nnUNet_results_analzation/scatter_plot_PETvsPETDWI.png')
