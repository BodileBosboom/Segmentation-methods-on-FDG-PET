import os
import nibabel as nib
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import csv
import glob

# def get_patient_numbers_from_labelmaps(base_path):
#     """
#     Automatically detects all patient numbers in the LabelMaps directory. Splits '_' from the filename since the paths have different file name structures
#     """
   
#     labelmaps_path = os.path.join(base_path, "LabelMaps")
#     patient_dirs = [d for d in os.listdir(labelmaps_path) if os.path.isdir(os.path.join(labelmaps_path, d))]
    
#     patient_numbers = []
#     for patient_dir in patient_dirs:
#         parts = patient_dir.split('_')
#         if len(parts) > 1:
#             patient_number = f"{parts[1]}_{parts[2]}" 
#             patient_numbers.append(patient_number)
    
#     return patient_numbers

def collect_paths_for_suv_calculation(base_path, patient_number,voxels):
    """
    Collects file paths for a given patient in the specified folders; LabelMaps, New LabelMaps and PET.

    """
    # Dictionary to hold paths for each file type
    file_paths = {}

    # Define the file patterns in each folder
    file_patterns = {
        "pet": os.path.join(base_path, "pet", f"ID{patient_number}*._pet_suv.nii"),
        "labelmaps": os.path.join(base_path, "LabelMaps", f"ID_{patient_number}", f"ID_{patient_number}_*label.nii"),
        "new_labelmaps": os.path.join(base_path, f"New_FSV_labelmaps_{voxels}", f"new_FSV_ID_{patient_number}_*label.nii")
    }

    # Find the first match for each file pattern except for new labelmaps
    for key, pattern in file_patterns.items():
        matching_files = glob.glob(pattern)
        if key == "labelmaps" or key == 'new_labelmaps':
            # For LabelMaps, we collect all matches as a list
            if matching_files:
                file_paths[key] = matching_files  # Store all matching LabelMaps
            else:
                print(f"Warning: No labelmap files found for pattern '{pattern}'")
                file_paths[key] = []
        else:
            file_paths[key] = matching_files[0] if matching_files else None

    print(matching_files)
    print(file_paths)

    return file_paths


def SUV_peak(convolved):
    max_value = np.nanmax(convolved)
    max_position = tuple(int(i) for i in np.unravel_index(np.nanargmax(convolved), convolved.shape))
    return max_value, max_position

def calculates_SUV_of_FSV(pet_path, fsv_path, fsv_erosion_path):
    PET_image = nib.load(pet_path).get_fdata()
    FSV_image = nib.load(fsv_path).get_fdata()
    FSV_erosion = nib.load(fsv_erosion_path).get_fdata()

    mask, mask_eroded = (FSV_image == 1), (FSV_erosion == 1)
    suv_within_voi, suv_within_voi_eroded = np.full_like(PET_image, np.nan), np.full_like(PET_image, np.nan)
    suv_within_voi[mask], suv_within_voi_eroded[mask_eroded] = PET_image[mask], PET_image[mask_eroded]

    suv_max, suv_max_position = np.nanmax(suv_within_voi), tuple(int(i) for i in np.unravel_index(np.nanargmax(suv_within_voi), suv_within_voi.shape))
    suv_max_eroded, suv_max_eroded_position = np.nanmax(suv_within_voi_eroded), tuple(int(i) for i in np.unravel_index(np.nanargmax(suv_within_voi_eroded), suv_within_voi_eroded.shape))
    suv_mean, suv_mean_eroded = np.nanmean(suv_within_voi), np.nanmean(suv_within_voi_eroded)

    structuring_element_3d = np.array([
        [[0.21, 0.56, 0.21], [0.56, 0.98, 0.56], [0.21, 0.56, 0.21]],
        [[0.56, 0.98, 0.56], [0.98, 1.00, 0.98], [0.56, 0.98, 0.56]],
        [[0.21, 0.56, 0.21], [0.56, 0.98, 0.56], [0.21, 0.56, 0.21]]
    ], dtype=np.float32)

    convolved = convolve(suv_within_voi, structuring_element_3d, mode='constant', cval=np.nan) / 15.28
    convolved_eroded = convolve(suv_within_voi_eroded, structuring_element_3d, mode='constant', cval=np.nan) / 15.28

    suv_peak, suv_peak_position = SUV_peak(convolved)
    suv_peak_eroded, suv_peak_position_eroded = SUV_peak(convolved_eroded)

    return [suv_max, suv_max_position, suv_mean, suv_peak, suv_peak_position, suv_max_eroded, suv_max_eroded_position, suv_mean_eroded, suv_peak_eroded, suv_peak_position_eroded]


# def process_SUV_calculations_and_save_to_csv_file(base_path):

#     results_all_patients=[]

#     patient_numbers = get_patient_numbers_from_labelmaps(base_path)

#     # Process each patient number
#     for patient_number in patient_numbers:
#         print(f"Processing patient with ID number {patient_number}")
#         patient_results = calculate_SUV_for_every_labelmap(base_path, patient_number)
#         results_all_patients.extend(patient_results)

#     # Define the output path for the aggregated CSV file
#     output_directory = os.path.join(base_path, 'SUV_values')
#     os.makedirs(output_directory, exist_ok=True)
#     output_path = os.path.join(output_directory, f'SUV_values.csv')

#     # Write results to a CSV file
#     with open(output_path, mode='w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Patient ID', 'Joint Name', 'SUVmax', 'SUVmax position', 'SUVmax_eroded', 'SUVmax_eroded position', 'SUVmean', 'SUVmean_eroded', 'SUVpeak', 'suv_peak_position', 'SUVpeak_eroded', 'suv_peak_eroded_position'])
#         writer.writerows(results_all_patients)

#     print(f"Results saved to {output_path}")

def calculate_SUV_for_every_labelmap(base_path, patient_number,voxels):
    """

    """
    # Collect all paths for the patient
    paths = collect_paths_for_suv_calculation(base_path, patient_number, voxels)

    # Retrieve paths for other required files
    pet_path = paths["pet"]
    fsv_path = paths["labelmaps"]
    fsv_erosion_path = paths["new_labelmaps"]

    results = []
    # keywords = ['shoulder','elbow','wrist','hip', 'knee','ankle','left','right', '60mm', '70mm', '80mm','90mm','100mm']

    # if pet_path and fsv_paths and fsv_erosion_paths:
    #     # Maak dictionaries op basis van bestandsnamen zonder '-label.nii'
    #     fsv_dict = {os.path.basename(f).replace('-label.nii', ''): f for f in fsv_paths}
    #     fsv_erosion_dict = {os.path.basename(f).replace('-label.nii', ''): f for f in fsv_erosion_paths}

    #     # Loop door de dictionary en koppel fsv_path en fsv_erosion_path op basis van naam
    #     for label_name, fsv_path in fsv_dict.items():
    #         if label_name in fsv_erosion_dict:
    #             fsv_erosion_path = fsv_erosion_dict[label_name]

    #             # Bereken SUV-waarden
    #             suv_values = calculates_SUV_of_FSV(pet_path, fsv_path, fsv_erosion_path)

    #             # Voeg de gefilterde woorden aan de resultaten toe
    #             words_in_basename = label_name.split('_')
    #             filtered_words = [word for word in words_in_basename if word in keywords]
    #             filtered_labelmap_basename = '_'.join(filtered_words) + f"_{voxels}"
                
    #             # Resultaat voor deze labelmap toevoegen
    #             results.append([patient_number] + [filtered_labelmap_basename] + suv_values)

    #         else:
    #             print(f"Waarschuwing: Geen overeenkomende file gevonden in 'new_labelmaps' voor {label_name}")

    if pet_path and fsv_path and fsv_erosion_path:
        for fsv_path, fsv_erosion_path in zip(fsv_path, fsv_erosion_path):
            # Calculate SUV values
            suv_values = calculates_SUV_of_FSV(pet_path, fsv_path, fsv_erosion_path)

            labelmap_basename = os.path.basename(fsv_path).replace('-label.nii', '').replace(f"ID_{patient_number}_","")
            # words_in_basename = labelmap_basename.split('_')
            # filtered_words = [word for word in words_in_basename if word in keywords]
            # filtered_labelmap_basename = '_'.join(filtered_words) + f"_{voxels}"
            # results.append([patient_number]+ [filtered_labelmap_basename] + suv_values)
            results.append([patient_number]+ [labelmap_basename] + suv_values)
    else:
        print("Missing required paths: Ensure pet_path, fsv_path, and fsv_erosion_path are provided.")
        

    return results