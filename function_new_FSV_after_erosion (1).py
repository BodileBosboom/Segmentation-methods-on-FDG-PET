"""
Returns new FSVs after erosion for all labelmaps 

"""

import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

def get_patient_numbers_from_labelmaps(base_path):
    """
    Automatically detects all patient numbers in the LabelMaps directory.
    """
   
    labelmaps_path = os.path.join(base_path, "LabelMaps")
    patient_dirs = [d for d in os.listdir(labelmaps_path) if os.path.isdir(os.path.join(labelmaps_path, d))]
    
    patient_numbers = []
    for patient_dir in patient_dirs:
        parts = patient_dir.split('_')
        if len(parts) > 1:
            patient_number = f"{parts[1]}_{parts[2]}" 
            patient_numbers.append(patient_number)
    
    return patient_numbers

def collect_paths(base_path, patient_number):
    """
    Collects file paths for a given patient in the specified folders.
    
    Parameters:
        base_path (str): Base directory containing all folders (ct, LabelMap, seg_bones, seg).
        patient_number (str): Patient ID to search for.
        
    Returns:
        dict: Dictionary with keys as file categories and values as file paths.
              The "labelmaps" key contains a list of paths to all LabelMaps for the patient.
    """

    # Dictionary to hold paths for each file type
    file_paths = {}
    
    # Define the file patterns in each folder
    file_patterns = {
        "ct": os.path.join(base_path, "ct", f"ID{patient_number}*._ct.nii"),
        "labelmaps": os.path.join(base_path, "LabelMaps", f"ID_{patient_number}", f"ID_{patient_number}_*label.nii"),
        "seg1": os.path.join(base_path, "seg_bones", f"seg_bones_ID{patient_number}*._ct.nii"),
        "seg2": os.path.join(base_path, "seg", f"seg_ID{patient_number}*._ct.nii")
    }

    # Find the first match for each file pattern except for labelmaps
    for key, pattern in file_patterns.items():
        if key == "labelmaps":
            # For LabelMaps, we collect all matches as a list
            matching_files = glob.glob(pattern)
            print(matching_files)
            if matching_files:
                file_paths[key] = matching_files  # Store all matching LabelMaps
            else:
                print(f"Warning: No labelmap files found for pattern '{pattern}'")
                file_paths[key] = []
        else:
            # For other files, just find the first match
            matching_files = glob.glob(pattern)
            file_paths[key] = matching_files[0] if matching_files else None

            print(file_paths)


    return file_paths

def perform_erosion_on_segmentations(seg1_path, seg2_path, voxels):
    """
    Perform erosion on the segmentation maps once and return the eroded versions.
    """
    # Load the segmentation images
    segmentation1_image = nib.load(seg1_path)
    segmentation2_image = nib.load(seg2_path)

    data_segmentation1 = segmentation1_image.get_fdata()
    data_segmentation2 = segmentation2_image.get_fdata()

    # Erosion of segmentation maps
    structuring_element_3d = np.ones((voxels, voxels, voxels), dtype=np.uint8)

    eroded_data_segmentation1 = np.zeros_like(data_segmentation1, dtype=np.uint8)
    eroded_data_segmentation2 = np.zeros_like(data_segmentation2, dtype=np.uint8)

    unique_structures = np.unique(np.concatenate((data_segmentation1[data_segmentation1 != 0], data_segmentation2[data_segmentation2 != 0])))

    for structure_label in unique_structures:
        binary_mask1 = (data_segmentation1 == structure_label).astype(np.uint8)
        eroded_mask1 = binary_erosion(binary_mask1, structuring_element_3d)
        eroded_data_segmentation1[eroded_mask1] = structure_label

        binary_mask2 = (data_segmentation2 == structure_label).astype(np.uint8)
        eroded_mask2 = binary_erosion(binary_mask2, structuring_element_3d)
        eroded_data_segmentation2[eroded_mask2] = structure_label

    return eroded_data_segmentation1, eroded_data_segmentation2

def create_new_FSV_with_erosion(ct_path, fsv_path, seg1_path, seg2_path, eroded_data_segmentation1, eroded_data_segmentation2, patient_number, voxels):
    """
    Create a new FSV by subtracting the segmentation maps of the bones after erosion.
    """

    # Upload files
    CT_image = nib.load(ct_path)
    FSV_image = nib.load(fsv_path)
   
    # Convert the images to NumPy arrays
    data_CT = CT_image.get_fdata()
    data_FSV = FSV_image.get_fdata()

    # Create binary masks
    threshold = 0  # Adjust this threshold as needed

    # Binary thresholding for FSV and segmentation maps
    threshold = 0
    binary_FSV = (data_FSV > threshold).astype(np.uint8)
    binary_eroded_data_segmentation1 = (eroded_data_segmentation1 > threshold).astype(np.uint8)
    binary_eroded_data_segmentation2 = (eroded_data_segmentation2 > threshold).astype(np.uint8)

    #test the erosion
    segmentation1_image = nib.load(seg1_path)
    segmentation2_image = nib.load(seg2_path)
    #load seg1path and seg2path 

    data_segmentation1 = segmentation1_image.get_fdata()
    data_segmentation2 = segmentation2_image.get_fdata()

    binary_segmentation1 = (data_segmentation1 > threshold).astype(np.uint8)  
    binary_segmentation2 = (data_segmentation2 > threshold).astype(np.uint8)  
    test1 = binary_segmentation1 - binary_eroded_data_segmentation1
    test2 = binary_segmentation2 - binary_eroded_data_segmentation2

    # Save test2
    new_test_2_image = nib.Nifti1Image(test2, FSV_image.affine, FSV_image.header)

    output_path = os.path.join(r'c:\Data TM-stage', f'segmenetation_maps', f'new_seg_.nii')
    print(output_path)
    
    # Create the output path using the patient number
    output_directory = os.path.dirname(output_path)    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(new_test_2_image, output_path)
    # print(f"New FSV saved to {output_path}")

    #Visualisation of the erosion
    ###for visualisation of the procedure at the slice with the maximum diameter of the FSV
    diameters = []

    # Loop through the slices and calculate the diameter
    for i in range(data_FSV.shape[1]):
        slice_data =data_FSV[:, i, :]
        diameter = np.sum(slice_data == 1)  # Count of voxels with value 1
        diameters.append(diameter)

    # Find the slice of the maximum diameter
    max_diameter_slice = np.argmax(diameters)

    # plt.figure(figsize=(15, 5))

    plt.subplot(2, 3, 1)  
    plt.imshow(np.flip(data_segmentation1[:, max_diameter_slice, :].swapaxes(0, 1)), cmap='gray')  # Coronal slice
    plt.title(f'Segmentation map 1 (slice {max_diameter_slice})')
    plt.axis('off')

    plt.subplot(2, 3, 2)  
    plt.imshow(np.flip(eroded_data_segmentation1[:, max_diameter_slice, :].swapaxes(0, 1)), cmap='gray')  # Coronal slice
    plt.title(f'Segmentation map 1 eroded (slice {max_diameter_slice})')
    plt.axis('off')

    plt.subplot(2, 3, 3)  
    plt.imshow(np.flip(test1[:, max_diameter_slice, :].swapaxes(0, 1)), cmap='gray') 
    plt.title(f'Erosion of segmentation map 1 (slice {max_diameter_slice})')
    plt.axis('off')

    plt.subplot(2, 3, 4)  
    plt.imshow(np.flip(data_segmentation2[:, max_diameter_slice, :].swapaxes(0, 1)), cmap='gray')  # Coronal slice
    plt.title(f'Segmentation map 2 (slice {max_diameter_slice})')
    plt.axis('off')

    plt.subplot(2, 3, 5)  
    plt.imshow(np.flip(eroded_data_segmentation2[:, max_diameter_slice, :].swapaxes(0, 1)), cmap='gray')  # Coronal slice
    plt.title(f'Segmentation map 2 eroded (slice {max_diameter_slice})')
    plt.axis('off')

    plt.subplot(2, 3, 6)  
    plt.imshow(np.flip(test2[:, max_diameter_slice, :].swapaxes(0, 1)), cmap='gray') 
    plt.title(f'Erosion of segmentation map 2 (slice {max_diameter_slice})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Create new FSV
    sum = binary_FSV + binary_eroded_data_segmentation1 + binary_eroded_data_segmentation2  
    mask = (sum == 2 | (sum == 3)).astype(np.uint8)  # Covers the case the segmentation maps overlap
    new_FSV = binary_FSV - mask

    # Create a figure with 3 subplots in a single row
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)  
    plt.imshow(np.flip(data_FSV[:, max_diameter_slice, :].swapaxes(0, 1)), cmap='gray')
    plt.title(f'FSV (slice {max_diameter_slice})')
    plt.axis('off')

    plt.subplot(1, 2, 2)  
    plt.imshow(np.flip(new_FSV[:, max_diameter_slice, :].swapaxes(0, 1)), cmap='gray')
    plt.title(f'New FSV (slice {max_diameter_slice})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Save the new FSV
    new_FSV_image = nib.Nifti1Image(new_FSV, FSV_image.affine, FSV_image.header)

    labelmap_basename = os.path.basename(fsv_path).replace('.nii', '')  # Get the base name of the LabelMap
    output_path = os.path.join(r'c:\Data TM-stage', f'New_FSV_labelmaps_{voxels}', f'new_FSV_{labelmap_basename}.nii')
    print(output_path)
    
    # Create the output path using the patient number
    output_directory = os.path.dirname(output_path)    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(new_FSV_image, output_path)
    # print(f"New FSV saved to {output_path}")

def create_new_FSV_after_erosion_for_every_labelmap(base_path, patient_number, voxels):
    """
    Uses funtions 'collect_paths' and 'create_new_FSV_with_erosion' to generate 
    file paths based on patient ID and create new FSV for every labelmap.
    
    Parameters:
        base_path (str): Base directory containing all folders.
        patient_number (str): Patient ID to search for.
    """
    # Collect all paths for the patient
    paths = collect_paths(base_path, patient_number)
    
    # Retrieve paths for other required files
    ct_path = paths["ct"]
    seg1_path = paths["seg1"]
    seg2_path = paths["seg2"]
    

    eroded_data_segmentation1, eroded_data_segmentation2 = perform_erosion_on_segmentations(seg1_path, seg2_path, voxels)
  
    # Check if labelmaps key exists and has files to process
    if "labelmaps" in paths and paths["labelmaps"]:
        # Loop through each LabelMap file and call the FSV creation function
        for fsv_path in paths["labelmaps"]:
            # print(f"Processing LabelMap: {fsv_path}")
            create_new_FSV_with_erosion(
                ct_path=ct_path,
                fsv_path=fsv_path,  
                seg1_path=seg1_path,
                seg2_path=seg2_path,
                eroded_data_segmentation1=eroded_data_segmentation1,
                eroded_data_segmentation2=eroded_data_segmentation2,
                patient_number=patient_number,
                voxels=voxels
            )

    else:
        print("No LabelMaps found for the given patient.")

# Get the list of patient numbers dynamically
base_path = r'c:\Data TM-stage'
voxels = 3
patient_numbers = get_patient_numbers_from_labelmaps(base_path)

# Loop over each patient number and process
for patient_number in patient_numbers:
    print(f"Processing patient with ID {patient_number}")
    create_new_FSV_after_erosion_for_every_labelmap(base_path, patient_number, voxels)
