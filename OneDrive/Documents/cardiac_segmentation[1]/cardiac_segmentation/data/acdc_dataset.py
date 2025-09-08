import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ACDCDataset(Dataset):
    """
    ACDC Cardiac MRI Dataset Handler
    
    Loads NIfTI files from ACDC dataset structure and handles 4-class segmentation:
    - Background (0)
    - Right Ventricle (1) 
    - Myocardium (2)
    - Left Ventricle (3)
    """
    
    def __init__(
        self,
        data_dir: str,
        patient_ids: List[str],
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = True,
        normalize: bool = True
    ):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        
        # Load all image and mask pairs
        self.data_pairs = self._load_data_pairs()
        
        # Setup transforms
        self.setup_transforms()
        
        logger.info(f"Loaded {len(self.data_pairs)} image-mask pairs from {len(patient_ids)} patients")
    
    def _load_data_pairs(self) -> List[Tuple[str, str]]:
        """Load all valid image-mask file pairs"""
        data_pairs = []
        
        for patient_id in self.patient_ids:
            patient_dir = os.path.join(self.data_dir, patient_id)
            
            if not os.path.exists(patient_dir):
                logger.warning(f"Patient directory not found: {patient_dir}")
                continue
            
            # Find all image files (both ED and ES timepoints)
            image_files = [f for f in os.listdir(patient_dir) 
                          if f.endswith('.nii.gz') and '_gt' not in f]
            
            for img_file in image_files:
                # Construct corresponding mask file name
                mask_file = img_file.replace('.nii.gz', '_gt.nii.gz')
                
                img_path = os.path.join(patient_dir, img_file)
                mask_path = os.path.join(patient_dir, mask_file)
                
                if os.path.exists(mask_path):
                    data_pairs.append((img_path, mask_path))
                else:
                    logger.warning(f"Mask not found for {img_file}")
        
        return data_pairs
    
    def setup_transforms(self):
        """Setup image transforms for preprocessing and augmentation"""
        transform_list = []
        
        if self.augment:
            # Basic augmentation: horizontal flip and rotation
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10, fill=0)
            ])
        
        # Always resize to target size
        transform_list.append(transforms.Resize(self.image_size))
        
        self.transform = transforms.Compose(transform_list)
    
    def _load_nifti(self, filepath: str) -> np.ndarray:
        """Load NIfTI file and return as numpy array"""
        try:
            nii = nib.load(filepath)
            data = nii.get_fdata()
            
            # Handle 3D volumes by selecting middle slice
            if len(data.shape) == 3:
                middle_slice = data.shape[2] // 2
                data = data[:, :, middle_slice]
            
            return data.astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image: normalize and convert to proper format"""
        # Normalize to [0, 1] if requested
        if self.normalize:
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        return image
    
    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Preprocess mask: ensure proper class labels"""
        # Ensure mask values are integers and within valid range
        mask = mask.astype(np.int64)
        mask = np.clip(mask, 0, 3)  # 4 classes: 0, 1, 2, 3
        
        return mask
    
    def __len__(self) -> int:
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get image-mask pair by index"""
        img_path, mask_path = self.data_pairs[idx]
        
        try:
            # Load image and mask
            image = self._load_nifti(img_path)
            mask = self._load_nifti(mask_path)
            
            # Preprocess
            image = self._preprocess_image(image)
            mask = self._preprocess_mask(mask)
            
            # Convert to PIL Images for transforms
            image_pil = transforms.ToPILImage()(image)
            mask_pil = transforms.ToPILImage()(mask.astype(np.uint8))
            
            # Apply same random transforms to both image and mask
            if self.transform:
                seed = torch.randint(0, 2**32, size=()).item()
                
                torch.manual_seed(seed)
                image_pil = self.transform(image_pil)
                
                torch.manual_seed(seed)
                mask_pil = self.transform(mask_pil)
            
            # Convert back to tensors
            image_tensor = transforms.ToTensor()(image_pil)
            mask_tensor = torch.from_numpy(np.array(mask_pil)).long()
            
            return image_tensor, mask_tensor
            
        except Exception as e:
            logger.error(f"Error processing pair {idx}: {img_path}, {mask_path}")
            logger.error(f"Error details: {str(e)}")
            # Return a dummy sample to avoid crashing
            dummy_image = torch.zeros((1, *self.image_size))
            dummy_mask = torch.zeros(self.image_size, dtype=torch.long)
            return dummy_image, dummy_mask


def create_data_splits(
    data_dir: str, 
    train_ratio: float = 0.8, 
    random_state: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Create train/validation splits from ACDC dataset
    
    Args:
        data_dir: Path to ACDC training directory
        train_ratio: Fraction of data for training
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_patient_ids, val_patient_ids)
    """
    # Find all patient directories
    patient_dirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('patient')]
    
    patient_dirs.sort()  # Ensure consistent ordering
    
    # Split patients (not individual images) to avoid data leakage
    train_patients, val_patients = train_test_split(
        patient_dirs, 
        train_size=train_ratio, 
        random_state=random_state,
        shuffle=True
    )
    
    logger.info(f"Data split: {len(train_patients)} train, {len(val_patients)} validation patients")
    
    return train_patients, val_patients


def create_data_loaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    image_size: Tuple[int, int] = (256, 256)
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation data loaders
    
    Args:
        data_dir: Path to ACDC training directory
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        train_ratio: Fraction of data for training
        image_size: Target image size (height, width)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create data splits
    train_patients, val_patients = create_data_splits(data_dir, train_ratio)
    
    # Create datasets
    train_dataset = ACDCDataset(
        data_dir=data_dir,
        patient_ids=train_patients,
        image_size=image_size,
        augment=True,
        normalize=True
    )
    
    val_dataset = ACDCDataset(
        data_dir=data_dir,
        patient_ids=val_patients,
        image_size=image_size,
        augment=False,  # No augmentation for validation
        normalize=True
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    data_dir = "path/to/ACDC/training"
    
    try:
        train_loader, val_loader = create_data_loaders(
            data_dir=data_dir,
            batch_size=4,
            num_workers=2
        )
        
        print(f"Train loader: {len(train_loader)} batches")
        print(f"Validation loader: {len(val_loader)} batches")
        
        # Test loading a batch
        for images, masks in train_loader:
            print(f"Image batch shape: {images.shape}")
            print(f"Mask batch shape: {masks.shape}")
            print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"Mask classes: {torch.unique(masks)}")
            break
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure ACDC dataset is available at the specified path")