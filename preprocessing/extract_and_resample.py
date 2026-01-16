import os
import SimpleITK as sitk
import pandas as pd
import numpy as np

def process_and_save_liver_only(patient_id, root_dir, output_dir, target_size=(256, 256, 160)):
    """
    Standardizes CT volumes by masking everything except the liver, 
    centering on the organ, and resampling to a uniform 1mm isotropic grid.
    """
    patient_folder = f"Lizard_ID{patient_id}"
    
    # 1. SET UP FOLDER REFERENCES
    image_dir = os.path.join(root_dir, patient_folder, f"STL_DICOM_{patient_folder}", "CorrespImage")
    mask_dir = os.path.join(root_dir, patient_folder, f"STL_DICOM_{patient_folder}", "Liver")
    
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"Skipping {patient_folder}: Missing Image or Mask folder.")
        return

    # 2. LOAD 3D VOLUMES
    reader = sitk.ImageSeriesReader()
    
    # Load CT Volume
    ct_names = reader.GetGDCMSeriesFileNames(image_dir)
    if not ct_names: return
    reader.SetFileNames(ct_names)
    ct_volume = reader.Execute()

    # Load Liver Mask Volume
    mask_names = reader.GetGDCMSeriesFileNames(mask_dir)
    if not mask_names: return
    reader.SetFileNames(mask_names)
    mask_volume = reader.Execute()

    # # 3. INTENSITY WINDOWING (LIVER WINDOW)
    # # Applying this before masking ensures the background is 0
    # window_filter = sitk.IntensityWindowingImageFilter()
    # window_filter.SetWindowMinimum(-100)
    # window_filter.SetWindowMaximum(250)
    # window_filter.SetOutputMinimum(0)
    # window_filter.SetOutputMaximum(255)
    # ct_volume = window_filter.Execute(ct_volume)

    # 4. VOXEL MASKING (ISOLATE LIVER)
    # Create a binary mask (ensuring only values 0 and 1)
    mask_binary = sitk.BinaryThreshold(mask_volume, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)
    
    # Multiply CT by Mask to zero-out everything else
    # We cast mask to float/int to match CT pixel type
    ct_masked = sitk.Multiply(ct_volume, sitk.Cast(mask_binary, ct_volume.GetPixelID()))

    # 5. FIND 3D CENTROID FOR STANDARDIZED CENTERING
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(mask_binary)
    
    available_labels = label_stats.GetLabels()
    if not available_labels:
        print(f"Skipping {patient_folder}: Mask is empty.")
        return
    
    target_label = 1 if 1 in available_labels else available_labels[0]
    liver_centroid = label_stats.GetCentroid(target_label) 

    # 6. ISOTROPIC RESAMPLING TO UNIFORM GRID
    # This forces the output to target_size (e.g., 256x256x160)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size) 
    resampler.SetOutputSpacing([1.0, 1.0, 1.0]) 
    resampler.SetOutputDirection(ct_masked.GetDirection())
    
    # Position the liver centroid in the middle of the new volume
    new_origin = [
        liver_centroid[i] - (target_size[i] * 1.0 / 2.0) for i in range(3)
    ]
    resampler.SetOutputOrigin(new_origin)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0) # All non-liver area remains black
    
    final_vol = resampler.Execute(ct_masked)

    # 7. SAVE AS NIFTI
    save_path = os.path.join(output_dir, f"{patient_folder}_liver_ONLY.nii.gz")
    sitk.WriteImage(final_vol, save_path)
    print(f"Saved Liver-Only volume: {save_path}")

# --- EXECUTION ---
csv_path = '/workspace/Storage_redundent/lizard/liver_slice_stats.csv'
root_data_path = "/workspace/Storage_fast/data/Mainz_LIZARD"
output_path = "/workspace/Storage_fast/data/Processed_Livers_Only"

os.makedirs(output_path, exist_ok=True)
df = pd.read_csv(csv_path) 

for _, row in df.iterrows():
    p_num = str(row['Patient_ID']).replace('Lizard_ID', '')
    try:
        process_and_save_liver_only(p_num, root_data_path, output_path)
    except Exception as e:
        print(f"Error on {p_num}: {e}")