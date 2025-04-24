import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from skimage import io, transform
import matplotlib.pyplot as plt
from torchvision import transforms

class PolypDataset(Dataset):
    """CVC-ClinicDB Polyp Dataset for RL-based segmentation"""

    def __init__(self, data_dir, transform=None, split='train', seed=42):
        """
        Args:
            data_dir (string): Directory with all the images and masks
            transform (callable, optional): Optional transforms to be applied on a sample
            split (string): 'train', 'val', or 'test'
            seed (int): Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.transform = transform
        self.seed = seed
        self.split = split
        
        # List all image files
        self.images_dir = os.path.join(data_dir, 'Original')
        self.masks_dir = os.path.join(data_dir, 'Ground Truth')
        
        self.images = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png') or f.endswith('.jpg')])
        
        print(f"Found {len(self.images)} images in {self.images_dir}")
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        
        # Split the dataset
        n_samples = len(self.images)
        indices = np.random.permutation(n_samples)
        
        if split == 'train':
            # 70% for training
            self.indices = indices[:int(0.7 * n_samples)]
        elif split == 'val':
            # 15% for validation
            self.indices = indices[int(0.7 * n_samples):int(0.85 * n_samples)]
        elif split == 'test':
            # 15% for testing
            self.indices = indices[int(0.85 * n_samples):]
        else:
            raise ValueError(f"Split {split} not recognized. Use 'train', 'val', or 'test'")
        
        self.image_files = [self.images[i] for i in self.indices]
        print(f"{split} split: {len(self.image_files)} images")
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        # Read image and mask
        try:
            image = io.imread(img_path)
            # Ensure image has 3 channels (RGB)
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=2)
                
            mask = io.imread(mask_path, as_gray=True)
            
            # Ensure mask is binary
            mask = (mask > 0).astype(np.float32)
            
            # Convert to torch tensors
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            
            sample = {'image': image, 'mask': mask, 'filename': img_name}
            
            if self.transform:
                sample = self.transform(sample)
                
            return sample
        except Exception as e:
            print(f"Error loading image {img_path} or mask {mask_path}: {e}")
            # Return a dummy sample if there's an error
            dummy_image = torch.zeros((3, 256, 256), dtype=torch.float32)
            dummy_mask = torch.zeros((1, 256, 256), dtype=torch.float32)
            return {'image': dummy_image, 'mask': dummy_mask, 'filename': img_name}
    
def get_data_loaders(data_dir, batch_size=8, num_workers=4, seed=42):
    """
    Create data loaders for train, validation, and test sets
    
    Args:
        data_dir (string): Directory with all the images and masks
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary with train, val, and test data loaders
    """
    # Define transformations
    transform = transforms.Compose([
        # Resize to fixed size
        ResizeTransform(target_size=(256, 256)),
    ])
    
    train_dataset = PolypDataset(data_dir=data_dir, transform=transform, split='train', seed=seed)
    val_dataset = PolypDataset(data_dir=data_dir, transform=transform, split='val', seed=seed)
    test_dataset = PolypDataset(data_dir=data_dir, transform=transform, split='test', seed=seed)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

class ResizeTransform:
    """Resize the image and mask to a fixed size"""
    
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        # Resize image and mask
        image = transforms.functional.resize(image, self.target_size)
        mask = transforms.functional.resize(mask, self.target_size)
        
        return {'image': image, 'mask': mask, 'filename': sample['filename']}

def dice_coefficient(y_pred, y_true, smooth=1e-6):
    """
    Calculate Dice Coefficient
    
    Args:
        y_pred (torch.Tensor): Predicted masks
        y_true (torch.Tensor): Ground truth masks
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        float: Dice coefficient
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    intersection = (y_pred * y_true).sum()
    dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    
    return dice

def iou_score(y_pred, y_true, smooth=1e-6):
    """
    Calculate IoU (Intersection over Union)
    
    Args:
        y_pred (torch.Tensor): Predicted masks
        y_true (torch.Tensor): Ground truth masks
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        float: IoU score
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou

def visualize_sample(sample, title=None):
    """
    Visualize a sample with its mask
    
    Args:
        sample (dict): Sample containing 'image' and 'mask'
        title (str, optional): Title for the plot
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(sample['image'], torch.Tensor):
        image = sample['image'].permute(1, 2, 0).cpu().numpy()
        
        # Rescale to 0-255 if normalized
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    else:
        image = sample['image']
    
    if isinstance(sample['mask'], torch.Tensor):
        mask = sample['mask'].squeeze().cpu().numpy()
    else:
        mask = sample['mask']
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.title('Overlay')
    plt.axis('off')
    
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    plt.show() 