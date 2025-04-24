import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import sys

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def extract_sample_image(data_dir='data/raw', sample_idx=100, output_dir='data/example_images'):
    """
    Extract a sample image and its mask from the dataset
    
    Args:
        data_dir (str): Path to dataset directory
        sample_idx (int): Index of the sample to extract
        output_dir (str): Directory to save extracted images
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    images_dir = os.path.join(data_dir, 'Original')
    masks_dir = os.path.join(data_dir, 'Ground Truth')
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png')])
    
    if sample_idx >= len(image_files):
        print(f"Sample index {sample_idx} is out of range. Using index 0 instead.")
        sample_idx = 0
    
    # Select a sample
    img_name = image_files[sample_idx]
    img_path = os.path.join(images_dir, img_name)
    mask_path = os.path.join(masks_dir, img_name)
    
    # Read image and mask
    image = io.imread(img_path)
    mask = io.imread(mask_path, as_gray=True)
    
    # Ensure mask is binary
    mask_binary = (mask > 0).astype(np.float32)
    
    # Save images
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask_binary, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(mask_binary, alpha=0.5, cmap='jet')
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_visualization.png'))
    plt.close()
    
    # Save individual images
    io.imsave(os.path.join(output_dir, 'original_image.png'), image)
    io.imsave(os.path.join(output_dir, 'ground_truth.png'), (mask_binary * 255).astype(np.uint8))
    
    # Create visualization with labels for proposal
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('(a) Colonoscopy image with polyp')
    plt.axis('off')
    
    # Ground truth mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask_binary, cmap='gray')
    plt.title('(b) Ground truth segmentation mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_example.png'), dpi=300)
    plt.close()
    
    print(f"Sample images extracted and saved to {output_dir}")
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask_binary.shape}")
    
    return image, mask_binary

if __name__ == "__main__":
    extract_sample_image() 