import os
import SimpleITK as sitk
import pandas as pd
import json
import re
import random

# 1. CONFIGURATION
EXCLUSION_LIST = ["115", "4", "13", "16", "26", "66", "69", "101", "146"]
TEST_SET_SIZE = 25
RANDOM_SEED = 42 
# INCREASED Z-DIMENSION to 256 for better vertical coverage
TARGET_SIZE = (256, 256, 256) 

def load_dicom_series(directory, reader):
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    if not dicom_names:
        raise FileNotFoundError(f"No DICOM files found in {directory}")
    reader.SetFileNames(dicom_names)
    return reader.Execute()

def resample_letterbox(img, is_label=False, target_size=TARGET_SIZE):
    target_spacing = [1.0, 1.0, 1.0]
    
    # Calculate physical center
    original_center = [
        img.GetOrigin()[i] + (img.GetSize()[i] * img.GetSpacing()[i] / 2.0)
        for i in range(3)
    ]
    
    # Calculate origin to center the liver in the larger 256-slice box
    new_origin = [
        original_center[i] - (target_size[i] * target_spacing[i] / 2.0)
        for i in range(3)
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(new_origin)
    resampler.SetOutputDirection(img.GetDirection())
    
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-100) # Background HU
        
    return resampler.Execute(img)

def process_patient(row, root_dir, target_raw_dir, is_test=False):
    patient_id = row['pid_str']
    p_folder = f"Lizard_ID{patient_id}"
    base_path = os.path.join(root_dir, p_folder, f"STL_DICOM_{p_folder}")
    
    img_sub = "imagesTs" if is_test else "imagesTr"
    lab_sub = "labelsTs" if is_test else "labelsTr"
    
    os.makedirs(os.path.join(target_raw_dir, img_sub), exist_ok=True)
    os.makedirs(os.path.join(target_raw_dir, lab_sub), exist_ok=True)

    reader = sitk.ImageSeriesReader()
    aligner = sitk.ResampleImageFilter()
    aligner.SetInterpolator(sitk.sitkNearestNeighbor)

    try:
        ct = load_dicom_series(os.path.join(base_path, "CorrespImage"), reader)
        aligner.SetReferenceImage(ct)
        
        # Merge Liver + Regions
        liver = aligner.Execute(load_dicom_series(os.path.join(base_path, "Liver"), reader))
        liver_mask = sitk.NotEqual(liver, 0)
        
        subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        region_folders = [f for f in subfolders if re.match(r'Region\d+', f, re.IGNORECASE)]
        for r_folder in region_folders:
            r_path = os.path.join(base_path, r_folder)
            r_mask = aligner.Execute(load_dicom_series(r_path, reader))
            liver_mask = sitk.Or(liver_mask, sitk.NotEqual(r_mask, 0))

        # Merge Vessels
        p_vol = aligner.Execute(load_dicom_series(os.path.join(base_path, "Portal"), reader))
        v_vol = aligner.Execute(load_dicom_series(os.path.join(base_path, "Vein"), reader))
        portal = sitk.Cast(sitk.NotEqual(p_vol, 0), sitk.sitkUInt8) * 1
        vein = sitk.Cast(sitk.NotEqual(v_vol, 0), sitk.sitkUInt8) * 2
        vessels = sitk.Maximum(portal, vein)

        # Intensity Clamping & Masking
        ct = sitk.Clamp(ct, sitk.sitkFloat32, -100, 250)
        ls = sitk.LabelShapeStatisticsImageFilter()
        ls.Execute(sitk.Cast(liver_mask, sitk.sitkUInt8))
        
        if ls.HasLabel(1):
            bbox = ls.GetBoundingBox(1)
            # Isolation
            m_float = sitk.Cast(liver_mask, sitk.sitkFloat32)
            ct = (ct * m_float) + ((1.0 - m_float) * -100.0)
        else:
            z_s, z_e = int(row['First_Liver_Slice']), int(row['Last_Liver_Slice'])
            bbox = [0, 0, z_s, ct.GetSize()[0], ct.GetSize()[1], max(1, z_e - z_s)]

        # ROI Crop
        sz = [min(ct.GetSize()[i] - bbox[i], bbox[i+3] + 4) for i in range(3)]
        idx = [max(0, bbox[i] - 2) for i in range(3)]
        
        # Resample to the new 256x256x256 Grid
        final_ct = resample_letterbox(sitk.RegionOfInterest(ct, sz, idx), False)
        final_lab = resample_letterbox(sitk.RegionOfInterest(vessels, sz, idx), True)

        sitk.WriteImage(final_ct, os.path.join(target_raw_dir, img_sub, f"Lizard_{patient_id}_0000.nii.gz"))
        sitk.WriteImage(final_lab, os.path.join(target_raw_dir, lab_sub, f"Lizard_{patient_id}.nii.gz"))
        return True
    except Exception as e:
        print(f"  --> Skip ID {patient_id}: {e}")
        return False

# --- EXECUTION ---
csv_path = '/workspace/Storage_redundent/lizard/stats/liver_slice.csv'
root_data = "/workspace/Storage_fast/data/Mainz_LIZARD"
raw_dir = "/workspace/Storage_fast/nnUNet_raw/Dataset501_LiverVessels"

df = pd.read_csv(csv_path)
df['pid_str'] = df['Patient_ID'].apply(lambda x: str(x).replace('Lizard_ID', ''))

# Filtering and Counting Usable Data
df_valid = df[~df['pid_str'].isin(EXCLUSION_LIST)].copy()
available_pids = [pid for pid in df_valid['pid_str'] if os.path.exists(os.path.join(root_data, f"Lizard_ID{pid}"))]
df_final = df_valid[df_valid['pid_str'].isin(available_pids)].copy()

total_usable = len(df_final)
print(f"Total Usable Patients: {total_usable}")

# Split
random.seed(RANDOM_SEED)
test_pids = random.sample(df_final['pid_str'].tolist(), min(TEST_SET_SIZE, total_usable))

success_count = 0
for _, row in df_final.iterrows():
    is_test = row['pid_str'] in test_pids
    if process_patient(row, root_data, raw_dir, is_test):
        success_count += 1

# Dataset.json update
dataset_json = {
    "channel_names": {"0": "CT"},
    "labels": {"background": 0, "portal_vein": 1, "hepatic_vein": 2},
    "numTrainingInstances": success_count - len(test_pids),
    "file_ending": ".nii.gz",
}
with open(os.path.join(raw_dir, "dataset.json"), 'w') as f:
    json.dump(dataset_json, f, indent=4)

print(f"Preprocessing finished. {success_count} patients saved in 256x256x256 grid.")