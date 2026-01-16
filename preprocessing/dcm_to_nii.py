#script to :
# 1. Convert dicom to nii.  
# 2. Organize files in nnunet format.

import os
import dicom2nifti
import dicom2nifti.settings as settings

settings.disable_validate_slicecount()

dataset_name = "Mainz_LIZARD"
data_root = "/workspace/Storage_fast/data/" + dataset_name + "/"
patient_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d.startswith("Lizard_ID")]

#create a new folder inside each patient folder to store nii files
for patient in patient_dirs:
    patient_path = os.path.join(data_root, patient)
    nii_output_path = os.path.join(patient_path, "NII_" + patient)
    os.makedirs(nii_output_path, exist_ok=True)

    # Convert DICOM to NIfTI
    dicom_base_input_path = os.path.join(patient_path, "STL_DICOM_" + patient)
    interested_anatomy = ("Vein", "Portal", "CorrespImage")
    anatomy_dirs = [d for d in os.listdir(dicom_base_input_path) if d.startswith(interested_anatomy)]
    for anatomy in anatomy_dirs:
        anatomy_dicom_path = os.path.join(dicom_base_input_path, anatomy)
        print(anatomy_dicom_path)
        os.makedirs(os.path.join(nii_output_path, anatomy), exist_ok=True)
        anatomy_nii_output_path = os.path.join(nii_output_path, anatomy + "/")
        dicom2nifti.convert_directory(anatomy_dicom_path, anatomy_nii_output_path, compression=True, reorient=True)
        print(anatomy_nii_output_path)
        print(f"Converted DICOM to NIfTI for {patient} - {anatomy} and saved to {anatomy_nii_output_path}")

    # dicom_input_path = os.path.join(dicom_base_input_path, "CorrespImage")
    # dcmtonii.convert_directory(dicom_input_path, nii_output_path)
    # print(f"Converted DICOM to NIfTI for {patient} and saved to {nii_output_path}")
    # print(f"Converted DICOM to NIfTI for {patient} and saved to {nii_output_path}")