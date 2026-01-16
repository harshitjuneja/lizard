import os
import SimpleITK as sitk
import pandas as pd
import json

def load_dicom_series(directory, reader):
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    if not dicom_names:
        raise FileNotFoundError(f"No DICOM files found in {directory}")
    reader.SetFileNames(dicom_names)
    return reader.Execute()

def resample_iso(img, is_label=False):
    target_sp = [1.0, 1.0, 1.0]
    orig_sp = img.GetSpacing()
    orig_sz = img.GetSize()
    new_sz = [int(round(orig_sz[i] * orig_sp[i] / target_sp[i])) for i in range(3)]
    
    res = sitk.ResampleImageFilter()
    res.SetSize(new_sz)
    res.SetOutputSpacing(target_sp)
    res.SetOutputOrigin(img.GetOrigin())
    res.SetOutputDirection(img.GetDirection())
    res.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return res.Execute(img)

def prepare_nnunet_vessels_universal(row, root_dir, target_raw_dir):
    patient_id = str(row['Patient_ID']).replace('Lizard_ID', '')
    p_folder = f"Lizard_ID{patient_id}"
    base_path = os.path.join(root_dir, p_folder, f"STL_DICOM_{p_folder}")
    
    reader = sitk.ImageSeriesReader()
    try:
        ct = load_dicom_series(os.path.join(base_path, "CorrespImage"), reader)
        liver = load_dicom_series(os.path.join(base_path, "Liver"), reader)
        portal = load_dicom_series(os.path.join(base_path, "Portal"), reader)
        vein = load_dicom_series(os.path.join(base_path, "Vein"), reader)
    except Exception as e:
        print(f"Skipping {p_folder}: {e}")
        return

    # 2. ALIGN PHYSICAL SPACES (Crucial for Patient 54 types)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    liver = resampler.Execute(liver)
    portal = resampler.Execute(portal)
    vein = resampler.Execute(vein)

    # This logic says: If it's not 0, it's a mask. 
    # This catches 1, 255, or any other value used during export.
    liver_mask = sitk.NotEqual(liver, 0)
    portal_mask = sitk.NotEqual(portal, 0)
    vein_mask = sitk.NotEqual(vein, 0)

    portal_lab = sitk.Cast(portal_mask, sitk.sitkUInt8) * 1
    vein_lab = sitk.Cast(vein_mask, sitk.sitkUInt8) * 2
    
    combined_labels = sitk.Maximum(portal_lab, vein_lab)

    ct = sitk.Clamp(ct, sitk.sitkFloat32, -100, 250)

    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(liver_mask)
    
    # Black out everything outside the liver
        
    if label_stats.HasLabel(1):
        bbox = label_stats.GetBoundingBox(1)
        liver_mask_float = sitk.Cast(liver_mask, sitk.sitkFloat32)
        ct_proc = (ct * liver_mask_float) + ((1.0 - liver_mask_float) * -100.0)
        buf = 5
    else:
        z_start = int(row['First_Liver_Slice'])
        z_end = int(row['Last_Liver_Slice'])
        bbox = [0, 0, z_start, ct.GetSize()[0], ct.GetSize()[1], max(1, z_end - z_start)]
        ct_proc = ct
        buf = 0

    size = [min(ct.GetSize()[i] - bbox[i], bbox[i+3] + 2*buf) for i in range(3)]
    index = [max(0, bbox[i] - buf) for i in range(3)]

    ct_crop = sitk.RegionOfInterest(ct_proc, size, index)
    lab_crop = sitk.RegionOfInterest(combined_labels, size, index)

    final_ct = resample_iso(ct_crop, is_label=False)
    final_lab = resample_iso(lab_crop, is_label=True)

    img_dir = os.path.join(target_raw_dir, "imagesTr")
    lab_dir = os.path.join(target_raw_dir, "labelsTr")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    
    sitk.WriteImage(final_ct, os.path.join(img_dir, f"Lizard_{patient_id}_0000.nii.gz"))
    sitk.WriteImage(final_lab, os.path.join(lab_dir, f"Lizard_{patient_id}.nii.gz"))
    print(f"Processed {p_folder}")

csv_path = '/workspace/Storage_redundent/lizard/stats/liver_slice.csv'
root_data = "/workspace/Storage_fast/data/Mainz_LIZARD"
raw_dir = "/workspace/Storage_fast/nnUNet_raw/Dataset501_LiverVessels"

df = pd.read_csv(csv_path)
for _, row in df.iterrows():
    prepare_nnunet_vessels_universal(row, root_data, raw_dir)