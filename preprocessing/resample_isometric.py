import SimpleITK as sitk
import os

def resample_to_1mm_isotropic(input_path, output_path):
    img = sitk.ReadImage(input_path)
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    target_spacing = [1.0, 1.0, 1.0]
    
    # 3. Calculate new size to maintain the same physical Field of View (FOV)
    # New Size = (Old Size * Old Spacing) / New Spacing
    
    target_size = [
        int(round(original_size[0] * original_spacing[0] / target_spacing[0])),
        int(round(original_size[1] * original_spacing[1] / target_spacing[1])),
        int(round(original_size[2] * original_spacing[2] / target_spacing[2]))
    ]
        
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    
    resampled_img = resampler.Execute(img)
    
    sitk.WriteImage(resampled_img, output_path)
    print(f"Resampled to: {target_size}")

# resample_to_1mm_isotropic("path/to/Corresplmage", "liver_iso.nii.gz")