import os
import pydicom
import numpy as np
import pandas as pd

def get_liver_stats(liver_dir):
    # Get all .dcm files and sort them to ensure correct slice order
    slices = sorted([f for f in os.listdir(liver_dir) if f.endswith('.dcm')])
    
    liver_indices = []
    total_liver_pixels_per_slice = []

    for idx, slice_name in enumerate(slices):
        ds = pydicom.dcmread(os.path.join(liver_dir, slice_name))
        pixel_array = ds.pixel_array
        
        # Check if the slice has any liver pixels (assuming mask > 0)
        num_pixels = np.count_nonzero(pixel_array)
        if num_pixels > 0:
            liver_indices.append(idx)
            total_liver_pixels_per_slice.append(num_pixels)

    if not liver_indices:
        return None  # No liver detected in any slice

    # 1. First and Last slice numbers
    first_slice = min(liver_indices)
    last_slice = max(liver_indices)

    # 2. Centroid Slice (Weighted average of slices based on liver volume/area)
    # This gives the slice that represents the 'center of mass' of the liver
    weights = np.array(total_liver_pixels_per_slice)
    centroid_slice = int(round(np.average(liver_indices, weights=weights)))

    return first_slice, last_slice, centroid_slice

# --- Main Batch Process ---
root_dir = "/workspace/Storage_fast/data/Mainz_LIZARD"
results = []

for patient_id in os.listdir(root_dir):
    patient_path = os.path.join(root_dir, patient_id)
    # Navigate to: Lizard_IDX / STL_DICOM_Lizard_IDX / Liver
    liver_path = os.path.join(patient_path, f"STL_DICOM_{patient_id}", "Liver")

    if os.path.exists(liver_path):
        print(f"Processing {patient_id}...")
        stats = get_liver_stats(liver_path)
        
        if stats:
            first, last, centroid = stats
            results.append({
                "Patient_ID": patient_id,
                "First_Liver_Slice": first,
                "Last_Liver_Slice": last,
                "Centroid_Slice": centroid,
                "Total_Slices_with_Liver": last - first + 1
            })

# Save results to a CSV for your analysis
df = pd.DataFrame(results)
df.to_csv("liver_slice_stats.csv", index=False)
print("Finished! Stats saved to liver_slice_stats.csv")