import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting, datasets, image
from nilearn.plotting import plot_stat_map
import os
from pathlib import Path
import pandas as pd
from scipy import ndimage
import seaborn as sns
import glob

class MeningiomaHeatmapGenerator:
    def __init__(self, output_dir='meningioma_analysis'):
        """
        Initialize the meningioma heatmap generator
        
        Parameters:
        -----------
        output_dir : str
            Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load MNI template
        self.mni_template = datasets.load_mni152_template(resolution=2)
        # Save the MNI template for reference
        template_path = self.output_dir / 'mni152_template.nii.gz'
        nib.save(self.mni_template, template_path)

    def register_to_mni(self, patient_image_path, patient_mask_path, 
                       registration_method='ants'):
        """
        Register patient image and mask to MNI space
        
        Parameters:
        -----------
        patient_image_path : str
            Path to patient T1 image
        patient_mask_path : str
            Path to patient meningioma segmentation mask
        registration_method : str
            Registration method ('ants' or 'fsl')
            
        Returns:
        --------
        registered_mask : nibabel image
            Mask registered to MNI space
        """
        
        if registration_method == 'ants':
            # Using ANTs registration (requires ANTsPy)
            try:
                import ants
                
                # Load patient images
                patient_img = ants.image_read(patient_image_path)
                mask_img = ants.image_read(patient_mask_path)
                
                # Create ANTs image from MNI template
                mni_data = self.mni_template.get_fdata()
                
                mni_img = ants.from_numpy(
                    mni_data,
                    origin=tuple(self.mni_template.affine[:3, 3]),
                    spacing=tuple(np.abs(np.diag(self.mni_template.affine[:3, :3]))),
                    direction=np.eye(3)
                )
                
                # Register patient T1 to MNI
                registration = ants.registration(
                    fixed=mni_img,
                    moving=patient_img,
                    type_of_transform='SyN'
                )
                
                # Apply transform to mask
                registered_mask = ants.apply_transforms(
                    fixed=mni_img,
                    moving=mask_img,
                    transformlist=registration['fwdtransforms'],
                    interpolator='nearestNeighbor'
                )
                
                # Convert back to nibabel
                registered_mask_nib = nib.Nifti1Image(
                    registered_mask.numpy(), 
                    affine=self.mni_template.affine
                )
                
                return registered_mask_nib
                
            except ImportError:
                print("ANTsPy not installed. Using nilearn registration instead.")
                return self._register_with_nilearn(patient_image_path, patient_mask_path)
            except Exception as e:
                print(f"ANTs registration failed: {e}")
                print("Falling back to nilearn registration.")
                return self._register_with_nilearn(patient_image_path, patient_mask_path)
                
        else:
            return self._register_with_nilearn(patient_image_path, patient_mask_path)
    
    def _register_with_nilearn(self, patient_image_path, patient_mask_path):
        """
        Fallback registration using nilearn
        """
        from nilearn.image import resample_to_img
        
        # Load patient images
        patient_img = nib.load(patient_image_path)
        mask_img = nib.load(patient_mask_path)
        
        # Simple resampling to MNI space (for demonstration)
        # In practice, you'd want proper affine registration
        registered_mask = resample_to_img(
            mask_img, 
            self.mni_template, 
            interpolation='nearest'
        )
        
        return registered_mask
    
    def register_to_mni_simple(self, patient_image_path, patient_mask_path):
        """
        Simple registration by resampling to MNI space
        This is faster but less accurate than full registration
        Use this for quick prototyping or when images are already roughly aligned
        """
        from nilearn.image import resample_to_img
        
        # Load mask
        mask_img = nib.load(patient_mask_path)
        
        # Simple resampling to MNI template space
        registered_mask = resample_to_img(
            mask_img, 
            self.mni_template, 
            interpolation='nearest'
        )
        
        return registered_mask
    
    def create_frequency_map(self, patient_data_list, registration_method='simple'):
        """
        Create frequency heatmap from multiple patient segmentations
        
        Parameters:
        -----------
        patient_data_list : list of tuples
            List of (patient_image_path, patient_mask_path) tuples
        registration_method : str
            Registration method ('simple', 'ants', or 'nilearn')
            
        Returns:
        --------
        frequency_map : nibabel image
            Frequency heatmap in MNI space
        """
        
        # Initialize frequency array
        template_shape = self.mni_template.shape
        frequency_array = np.zeros(template_shape)
        
        total_patients = len(patient_data_list)
        
        print(f"Processing {total_patients} patients using {registration_method} registration...")
        
        for i, (img_path, mask_path) in enumerate(patient_data_list):
            print(f"Processing patient {i+1}/{total_patients}")
            
            try:
                # Register mask to MNI space
                if registration_method == 'simple':
                    registered_mask = self.register_to_mni_simple(img_path, mask_path)
                else:
                    registered_mask = self.register_to_mni(img_path, mask_path, registration_method)
                
                # Add to frequency map
                mask_data = registered_mask.get_fdata()
                frequency_array += (mask_data > 0).astype(float)
                
            except Exception as e:
                print(f"Error processing patient {i+1}: {e}")
                print("Continuing with next patient...")
                continue
        
        # Convert to percentage
        frequency_percentage = (frequency_array / total_patients) * 100
        
        # Create nibabel image
        frequency_map = nib.Nifti1Image(
            frequency_percentage,
            affine=self.mni_template.affine
        )
        
        return frequency_map
    
    
def create_patient_data_list(base_directory, t1_pattern="*T1Post.nii.gz", mask_pattern="*abnormal_seg.nii.gz"):
    """
    Create patient data list from output directory structure
    
    Parameters:
    -----------
    base_directory : str
        Base directory containing patient folders (e.g., '/path/to/output/')
    t1_pattern : str
        Pattern to match T1 images (default: "*T1Post.nii.gz")
    mask_pattern : str
        Pattern to match segmentation masks (default: "*abnormal_seg.nii.gz")
    
    Returns:
    --------
    patient_data : list of tuples
        List of (t1_path, mask_path) tuples
    """
    
    base_path = Path(base_directory)
    patient_data = []
    missing_files = []
    
    # Get all patient directories (assuming they start with patient identifiers)
    patient_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    for patient_dir in sorted(patient_dirs):
        patient_id = patient_dir.name
        
        # Look for T1Post directory
        t1post_dir = patient_dir / "T1Post"
        
        if not t1post_dir.exists():
            print(f"Warning: No T1Post directory found for {patient_id}")
            missing_files.append(f"{patient_id}: No T1Post directory")
            continue
        
        # Find T1 image
        t1_files = list(t1post_dir.glob(t1_pattern))
        
        if not t1_files:
            print(f"Warning: No T1 image found for {patient_id}")
            missing_files.append(f"{patient_id}: No T1 image matching {t1_pattern}")
            continue
        
        if len(t1_files) > 1:
            print(f"Warning: Multiple T1 images found for {patient_id}, using first one")
        
        t1_path = str(t1_files[0])
        
        # Find segmentation mask in abnormalmap directory
        abnormalmap_dir = t1post_dir / "abnormalmap"
        
        if not abnormalmap_dir.exists():
            print(f"Warning: No abnormalmap directory found for {patient_id}")
            missing_files.append(f"{patient_id}: No abnormalmap directory")
            continue
        
        mask_files = list(abnormalmap_dir.glob(mask_pattern))
        
        if not mask_files:
            print(f"Warning: No segmentation mask found for {patient_id}")
            missing_files.append(f"{patient_id}: No mask matching {mask_pattern}")
            continue
        
        if len(mask_files) > 1:
            print(f"Warning: Multiple masks found for {patient_id}, using first one")
        
        mask_path = str(mask_files[0])
        
        # Add to patient data list
        patient_data.append((t1_path, mask_path))
        print(f"Added {patient_id}: T1={Path(t1_path).name}, Mask={Path(mask_path).name}")
    
    print(f"\nSuccessfully loaded {len(patient_data)} patients")
    
    if missing_files:
        print(f"\nMissing files for {len(missing_files)} patients:")
        for missing in missing_files:
            print(f"  - {missing}")
    
    return patient_data

def main():
    """
    Main function to run the meningioma heatmap analysis
    
    Usage:
    - Set the base_directory to your output directory
    - Run the script
    """
    
    # Initialize generator
    generator = MeningiomaHeatmapGenerator()
    
    # ==========================================
    # SET YOUR BASE DIRECTORY HERE
    # ==========================================
    base_directory = "/path/to/your/output/"  # Update this path
    
    # Validate that base directory exists
    if not Path(base_directory).exists():
        print(f"Error: Base directory '{base_directory}' does not exist.")
        print("Please update the base_directory variable in the main() function.")
        return
    
    # Create patient data list
    print("Scanning for patient data...")
    patient_data = create_patient_data_list(base_directory)
    
    if not patient_data:
        print("No valid patient data found. Please check your directory structure and file patterns.")
        return
    
    # Create frequency heatmap
    print("Creating frequency heatmap...")
    frequency_map = generator.create_frequency_map(patient_data, registration_method='simple')
    
    # Save frequency map
    print("Saving frequency map...")
    nib.save(frequency_map, generator.output_dir / 'frequency_map.nii.gz')
    
    print("Analysis complete!")
    print(f"Output files saved to: {generator.output_dir}")

if __name__ == "__main__":
    main()