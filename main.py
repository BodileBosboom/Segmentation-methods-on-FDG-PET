import os
import nibabel as nib
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import csv
import glob

from function_new_FSV_after_erosion import create_new_FSV_after_erosion_for_every_labelmap, get_patient_numbers_from_labelmaps
from suv_calculation_complete import calculate_SUV_for_every_labelmap

results_all_patients=[]


# Get the list of patient numbers dynamically
base_path = r'c:\Data TM-stage'
patient_numbers = get_patient_numbers_from_labelmaps(base_path)
voxels = 3

# Loop over each patient number and process
for patient_number in patient_numbers:
    print(f"Processing patient with ID {patient_number}")

    #Create new FSV after erosion
    create_new_FSV_after_erosion_for_every_labelmap(base_path, patient_number, voxels)

    # Calculate the SUV values for the current patient and voxel size
    patient_results = calculate_SUV_for_every_labelmap(base_path, patient_number, voxels)
    # Store the results in the list
    results_all_patients.extend(patient_results)        

# Define the output path for the aggregated CSV file
output_directory = os.path.join(base_path, 'SUV_values')
os.makedirs(output_directory, exist_ok=True)
output_path = os.path.join(output_directory, f'SUV_values_erosion_{voxels}_.csv')

# Write results to a CSV file
with open(output_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Patient ID', 'Joint Name', 'SUVmax', 'SUVmax position', 'SUVmean', 'SUVpeak', 'suv_peak_position', f'SUVmax_eroded_{voxels}', 
                     f'SUVmax_eroded_{voxels} position', f'SUVmean_eroded_{voxels}', f'SUVpeak_eroded_{voxels}', f'suv_peak_eroded_{voxels} position'])
    writer.writerows(results_all_patients)
