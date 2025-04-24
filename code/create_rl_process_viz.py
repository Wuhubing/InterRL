import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import matplotlib.patches as patches

def main():
    # Set directories
    output_dir = 'data/example_images'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load example image and mask
    image_path = os.path.join(output_dir, 'original_image.png')
    mask_path = os.path.join(output_dir, 'ground_truth.png')
    
    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path)) > 0
    
    # Image dimensions
    height, width, _ = image.shape
    
    # Create a visualization of the RL process
    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    
    # Step 1: Initial state (empty mask)
    empty_mask = np.zeros_like(mask)
    axes[0].imshow(image)
    axes[0].scatter(width//2, height//2, c='red', marker='x', s=100)
    axes[0].set_title('Step 1: Initial State\nEmpty mask, pointer at center')
    axes[0].axis('off')
    
    # Step 2: Agent moves pointer to region of interest
    axes[1].imshow(image)
    # Find a point inside the mask
    y, x = np.where(mask)
    if len(y) > 0:
        # Choose a point near the center of the mask
        idx = len(y) // 2
        ptr_y, ptr_x = y[idx], x[idx]
    else:
        # Fallback to image center
        ptr_y, ptr_x = height//2, width//2
    axes[1].scatter(ptr_x, ptr_y, c='red', marker='x', s=100)
    axes[1].set_title('Step 2: Move Pointer\nNavigate to region of interest')
    axes[1].axis('off')
    
    # Step 3: Agent expands region
    mask_partial = np.zeros_like(mask)
    # Create a small circular region around the pointer
    y_grid, x_grid = np.ogrid[-ptr_y:height-ptr_y, -ptr_x:width-ptr_x]
    mask_partial[(y_grid**2 + x_grid**2) <= 20**2] = 1
    
    axes[2].imshow(image)
    axes[2].imshow(mask_partial, alpha=0.5, cmap='jet')
    axes[2].scatter(ptr_x, ptr_y, c='red', marker='x', s=100)
    axes[2].set_title('Step 3: Expand Region\nCreate initial segmentation')
    axes[2].axis('off')
    
    # Step 4: Agent refines the segmentation
    # Create a more refined mask that's closer to ground truth but not perfect
    dilated_mask = np.zeros_like(mask)
    y_grid, x_grid = np.ogrid[-ptr_y:height-ptr_y, -ptr_x:width-ptr_x]
    dilated_mask[(y_grid**2 + x_grid**2) <= 30**2] = 1
    
    axes[3].imshow(image)
    axes[3].imshow(dilated_mask, alpha=0.5, cmap='jet')
    axes[3].scatter(ptr_x+15, ptr_y-10, c='red', marker='x', s=100)
    axes[3].set_title('Step 4: Refine Segmentation\nExpand/shrink to match boundaries')
    axes[3].axis('off')
    
    # Step 5: Final segmentation
    axes[4].imshow(image)
    axes[4].imshow(mask, alpha=0.5, cmap='jet')
    axes[4].set_title('Step 5: Final Result\nConfirm segmentation')
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rl_process.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"RL process visualization saved to {os.path.join(output_dir, 'rl_process.png')}")

if __name__ == '__main__':
    main() 