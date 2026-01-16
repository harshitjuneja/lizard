import os
import SimpleITK as sitk
import pandas as pd

def process_nnunet_gold_standard(patient_id, root_dir, output_dir):
    """
    Standardizes CT volumes for nnU-Net:
    1. Aligns CT and Mask physical spaces.
    2. Clips intensity to Liver Window (-100 to 250 HU).
    3. Crops tightly to the liver bounding box.
    4. Masks out non-liver anatomy (fills background with -100 HU).
    5. Resamples to 1mm isotropic resolution.
    """
    patient_folder = f"Lizard_ID{patient_id}"
    image_dir = os.path.join(root_dir, patient_folder, f"STL_DICOM_{patient_folder}", "CorrespImage")
    mask_dir = os.path.join(root_dir, patient_folder, f"STL_DICOM_{patient_folder}", "Liver")
    
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"Skipping {patient_id}: Missing folder.")
        return

    # 1. LOAD VOLUMES
    reader = sitk.ImageSeriesReader()
    
    # Load CT
    ct_files = reader.GetGDCMSeriesFileNames(image_dir)
    if not ct_files: return
    reader.SetFileNames(ct_files)
    ct_vol = reader.Execute()

    # Load Mask
    mask_files = reader.GetGDCMSeriesFileNames(mask_dir)
    if not mask_files: return
    reader.SetFileNames(mask_files)
    mask_vol = reader.Execute()

    # 2. ALIGN PHYSICAL SPACES (Fixes the "Inputs do not occupy same physical space" error)
    # This ensures the mask matches the CT's grid exactly.
    if mask_vol.GetSpacing() != ct_vol.GetSpacing() or \
       mask_vol.GetOrigin() != ct_vol.GetOrigin() or \
       mask_vol.GetDirection() != ct_vol.GetDirection():
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ct_vol)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor) # Must be NN for masks
        resampler.SetTransform(sitk.Transform())
        mask_vol = resampler.Execute(mask_vol)

    # 3. INTENSITY CLAMPING (-100 to 250 HU)
    # Standard liver range; preserves raw HU values for nnU-Net.
    ct_vol = sitk.Clamp(ct_vol, sitk.sitkFloat32, lowerBound=-100, upperBound=250)

    # 4. GET TIGHT BOUNDING BOX FROM MASK
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(mask_vol)
    
    available_labels = label_stats.GetLabels()
    if not available_labels:
        print(f"Skipping {patient_id}: Mask appears empty.")
        return
    
    # Detect label (handles 1, 255, etc.)
    target_label = 1 if 1 in available_labels else available_labels[0]
    bbox = label_stats.GetBoundingBox(target_label)
    
    # 5. CROP BOTH CT AND MASK
    # Buffer of 3 voxels to provide a tiny bit of context
    buffer = 3
    size = [min(ct_vol.GetSize()[i] - bbox[i], bbox[i+3] + 2*buffer) for i in range(3)]
    index = [max(0, bbox[i] - buffer) for i in range(3)]
    
    ct_crop = sitk.RegionOfInterest(ct_vol, size, index)
    mask_crop = sitk.RegionOfInterest(mask_vol, size, index)

    # 6. MASKING (Set non-liver areas to -100 HU)
    mask_binary = sitk.BinaryThreshold(mask_crop, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)
    ct_float = sitk.Cast(ct_crop, sitk.sitkFloat32)
    mask_float = sitk.Cast(mask_binary, sitk.sitkFloat32)
    
    # Logical Masking: (CT * Mask) + (InverseMask * -100)
    ct_masked = (ct_float * mask_float) + ((1.0 - mask_float) * -100.0)

    # 7. RESAMPLE TO 1mm ISOTROPIC
    # Prevents anatomy distortion and standardizes the "zoom" for the network.
    orig_spacing = ct_masked.GetSpacing()
    orig_size = ct_masked.GetSize()
    target_spacing = [1.0, 1.0, 1.0]
    new_size = [int(round(orig_size[i] * orig_spacing[i] / target_spacing[i])) for i in range(3)]

    iso_resampler = sitk.ResampleImageFilter()
    iso_resampler.SetSize(new_size)
    iso_resampler.SetOutputSpacing(target_spacing)
    iso_resampler.SetOutputOrigin(ct_masked.GetOrigin())
    iso_resampler.SetOutputDirection(ct_masked.GetDirection())
    iso_resampler.SetInterpolator(sitk.sitkLinear)
    
    final_vol = iso_resampler.Execute(ct_masked)

    # 8. SAVE IN nnU-Net FORMAT
    save_path = os.path.join(output_dir, f"Lizard_{patient_id}_0000.nii.gz")
    sitk.WriteImage(final_vol, save_path)
    print(f"Success: {patient_id} | Size: {final_vol.GetSize()}")

# --- EXECUTION ---
root_data = "/workspace/Storage_fast/data/Mainz_LIZARD"
out_data = "/workspace/Storage_fast/data/Processed_nnUNet"
csv_path = '/workspace/Storage_redundent/lizard/liver_slice_stats.csv'

os.makedirs(out_data, exist_ok=True)
df = pd.read_csv(csv_path)

print(f"Processing {len(df)} patients...")
for _, row in df.iterrows():
    p_num = str(row['Patient_ID']).replace('Lizard_ID', '')
    try:
        process_nnunet_gold_standard(p_num, root_data, out_data)
    except Exception as e:
        print(f"CRITICAL ERROR on {p_num}: {e}")

print("Preprocessing Complete.")