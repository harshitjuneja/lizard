import os
import SimpleITK as sitk
import pandas as pd

def process_and_save_liver(patient_id, first_slice, last_slice, root_dir, output_dir):
    patient_folder = f"Lizard_ID{patient_id}"
    image_dir = f"/workspace/Storage_fast/data/Mainz_LIZARD/{patient_folder}/STL_DICOM_{patient_folder}/CorrespImage"
    
    if not os.path.exists(image_dir):
        print(f"Directory not found for {patient_folder}")
        return

    # 2. Load DICOM Series as a 3D Image
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(image_dir)
    reader.SetFileNames(dicom_names)
    full_volume = reader.Execute()

    # 3. Perform Z-Axis Crop (with 1-slice buffer)
    z_min = max(0, first_slice - 1)
    z_max = min(full_volume.GetSize()[2] - 1, last_slice + 1)
    
    size = list(full_volume.GetSize())
    size[2] = z_max - z_min + 1
    index = [0, 0, z_min]
    cropped_vol = sitk.RegionOfInterest(full_volume, size, index)

    # 4. Isotropic Resampling to 1mm^3
    original_spacing = cropped_vol.GetSpacing()
    original_size = cropped_vol.GetSize()
    target_spacing = [1.0, 1.0, 1.0]

    new_size = [
        int(round(original_size[0] * original_spacing[0] / target_spacing[0])),
        int(round(original_size[1] * original_spacing[1] / target_spacing[1])),
        int(round(original_size[2] * original_spacing[2] / target_spacing[2]))
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(cropped_vol.GetOrigin())
    resampler.SetOutputDirection(cropped_vol.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    
    final_vol = resampler.Execute(cropped_vol)

    # 5. Saving Logic
    save_path = os.path.join(output_dir, f"{patient_folder}_liver_1mm.nii.gz")
    sitk.WriteImage(final_vol, save_path)
    print(f"Saved: {save_path}")

# --- Execution ---
# Assuming your CSV data is in a DataFrame called 'df'
df = pd.read_csv('/workspace/Storage_redundent/lizard/liver_slice_stats.csv') 
output_path = "/workspace/Storage_fast/data/Processed_Livers"
os.makedirs(output_path, exist_ok=True)

for _, row in df.iterrows():
    # Extract ID number from 'Lizard_ID195' -> 195
    p_num = str(row['Patient_ID']).replace('Lizard_ID', '')
    process_and_save_liver(
        p_num, 
        row['First_Liver_Slice'], 
        row['Last_Liver_Slice'], 
        "/workspace/Storage_fast/data/Mainz_LIZARD", 
        output_path
    )