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
                
                # Load images
                patient_img = ants.image_read(patient_image_path)
                mask_img = ants.image_read(patient_mask_path)
                mni_img = ants.image_read(self.mni_template.get_fdata())
                
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
    
    def create_frequency_map(self, patient_data_list):
        """
        Create frequency heatmap from multiple patient segmentations
        
        Parameters:
        -----------
        patient_data_list : list of tuples
            List of (patient_image_path, patient_mask_path) tuples
            
        Returns:
        --------
        frequency_map : nibabel image
            Frequency heatmap in MNI space
        """
        
        # Initialize frequency array
        template_shape = self.mni_template.shape
        frequency_array = np.zeros(template_shape)
        
        total_patients = len(patient_data_list)
        
        print(f"Processing {total_patients} patients...")
        
        for i, (img_path, mask_path) in enumerate(patient_data_list):
            print(f"Processing patient {i+1}/{total_patients}")
            
            # Register mask to MNI space
            registered_mask = self.register_to_mni(img_path, mask_path)
            
            # Add to frequency map
            mask_data = registered_mask.get_fdata()
            frequency_array += (mask_data > 0).astype(float)
        
        # Convert to percentage
        frequency_percentage = (frequency_array / total_patients) * 100
        
        # Create nibabel image
        frequency_map = nib.Nifti1Image(
            frequency_percentage,
            affine=self.mni_template.affine
        )
        
        return frequency_map
    
    def create_heatmap_visualization(self, frequency_map, output_filename='meningioma_heatmap.png'):
        """
        Create multi-slice heatmap visualization
        
        Parameters:
        -----------
        frequency_map : nibabel image
            Frequency heatmap
        output_filename : str
            Output filename for the visualization
        """
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Define slice positions for each view
        axial_slices = [-40, -20, 0, 20, 40, 60]
        sagittal_slices = [-40, -20, 0, 20, 40, 60]
        coronal_slices = [-60, -40, -20, 0, 20, 40]
        
        # Axial slices
        for i, z_coord in enumerate(axial_slices):
            ax = plt.subplot(3, 6, i+1)
            plotting.plot_stat_map(
                frequency_map,
                bg_img=self.mni_template,
                display_mode='z',
                cut_coords=[z_coord],
                colorbar=False,
                cmap='hot',
                vmax=5.0,
                axes=ax,
                title=f'Axial z={z_coord}'
            )
        
        # Sagittal slices
        for i, x_coord in enumerate(sagittal_slices):
            ax = plt.subplot(3, 6, i+7)
            plotting.plot_stat_map(
                frequency_map,
                bg_img=self.mni_template,
                display_mode='x',
                cut_coords=[x_coord],
                colorbar=False,
                cmap='hot',
                vmax=5.0,
                axes=ax,
                title=f'Sagittal x={x_coord}'
            )
        
        # Coronal slices
        for i, y_coord in enumerate(coronal_slices):
            ax = plt.subplot(3, 6, i+13)
            plotting.plot_stat_map(
                frequency_map,
                bg_img=self.mni_template,
                display_mode='y',
                cut_coords=[y_coord],
                colorbar=False,
                cmap='hot',
                vmax=5.0,
                axes=ax,
                title=f'Coronal y={y_coord}'
            )
        
        # Add colorbar
        plt.tight_layout()
        
        # Create colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(0, 5)),
            cax=cbar_ax
        )
        cbar.set_label('Tumor Frequency (%)', rotation=270, labelpad=20)
        
        plt.savefig(self.output_dir / output_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Heatmap saved to {self.output_dir / output_filename}")
    
    
# Example usage
def main():
    # Initialize generator
    generator = MeningiomaHeatmapGenerator()
    
    # Use your own patient data
    # patient_data = [
    #     ('patient1_T1.nii.gz', 'patient1_mask.nii.gz'),
    #     ('patient2_T1.nii.gz', 'patient2_mask.nii.gz'),
    #     # ... add more patients
    # ]
    
    
    # Create frequency heatmap
    print("Creating frequency heatmap...")
    frequency_map = generator.create_frequency_map(patient_data)
    
    # Save frequency map
    nib.save(frequency_map, generator.output_dir / 'frequency_map.nii.gz')
    
    # Create visualization
    print("Creating visualization...")
    generator.create_heatmap_visualization(frequency_map)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()