import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add code directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from code.data_utils import PolypDataset

def main():
    # Set data directory
    data_dir = 'data/raw'
    
    # Create dataset
    train_dataset = PolypDataset(data_dir=data_dir, transform=None, split='train', seed=42)
    
    print(f"Dataset loaded with {len(train_dataset)} samples")
    
    # Try to load a sample
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"Sample loaded with image shape: {sample['image'].shape} and mask shape: {sample['mask'].shape}")
        
        # Visualize sample
        plt.figure(figsize=(10, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(sample['image'].permute(1, 2, 0).numpy())
        plt.title('Image')
        plt.axis('off')
        
        # Ground truth mask
        plt.subplot(1, 3, 2)
        plt.imshow(sample['mask'].squeeze().numpy(), cmap='gray')
        plt.title('Mask')
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(sample['image'].permute(1, 2, 0).numpy())
        plt.imshow(sample['mask'].squeeze().numpy(), alpha=0.5, cmap='jet')
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('data/example_images/data_test.png')
        plt.close()
        
        print(f"Sample visualization saved to data/example_images/data_test.png")

if __name__ == '__main__':
    main() 