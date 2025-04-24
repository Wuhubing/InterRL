import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():
    # Set directories
    images_dir = 'data/raw/Original'
    masks_dir = 'data/raw/Ground Truth'
    output_dir = 'data/example_images'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png')])
    
    # Select a sample
    sample_idx = 50  # Choose a different index
    if sample_idx >= len(image_files):
        sample_idx = 0
    
    img_name = image_files[sample_idx]
    print(f"Selected image: {img_name}")
    
    img_path = os.path.join(images_dir, img_name)
    mask_path = os.path.join(masks_dir, img_name)
    
    # Read image and mask with PIL
    image = Image.open(img_path)
    mask = Image.open(mask_path).convert('L')
    
    # Convert to numpy arrays
    image_array = np.array(image)
    mask_array = np.array(mask)
    
    # Ensure mask is binary
    mask_binary = (mask_array > 0).astype(np.uint8) * 255
    
    # Save with PIL
    Image.fromarray(image_array).save(os.path.join(output_dir, 'original_image.png'))
    Image.fromarray(mask_binary).save(os.path.join(output_dir, 'ground_truth.png'))
    
    # Save visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(image_array)
    axes[0].set_title('(a) Colonoscopy image with polyp')
    axes[0].axis('off')
    
    axes[1].imshow(mask_binary, cmap='gray')
    axes[1].set_title('(b) Ground truth segmentation mask')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_example.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Images saved to {output_dir}")
    print(f"Image dimensions: {image_array.shape}")
    print(f"Mask dimensions: {mask_array.shape}")

if __name__ == '__main__':
    main() 