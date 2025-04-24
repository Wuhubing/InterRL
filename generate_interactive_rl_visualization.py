#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create visualization of the interactive RL segmentation process
Demonstrates how InteractiveRL progressively optimizes segmentation results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import cv2
from PIL import Image, ImageDraw, ImageFont

# Set matplotlib parameters for academic-style charts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Set base color scheme - academic colors
COLOR_ORIGINAL = '#1f77b4'  # Blue - original image
COLOR_MASK = '#ff7f0e'      # Orange - segmentation mask
COLOR_POINTER = '#2ca02c'   # Green - pointer location
COLOR_ACTION = '#d62728'    # Red - action

# Create output directory
output_dir = 'academic_figures'
os.makedirs(output_dir, exist_ok=True)

# Simulate states in the RL process
def generate_synthetic_rl_process():
    """
    Generate a simulated interactive RL segmentation process
    Returns a sequence of states at multiple timesteps
    """
    # Step 1: Create a synthetic colonoscopy image and polyp region
    img_size = (256, 256)
    # Create background - simulate pink background of colonoscopy images
    background = np.ones((img_size[0], img_size[1], 3), dtype=np.float32) * np.array([0.8, 0.7, 0.7])
    
    # Add some vessel-like textures
    for i in range(20):
        x1, y1 = np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1])
        x2, y2 = np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1])
        cv2.line(background, (x1, y1), (x2, y2), (0.7, 0.5, 0.5), thickness=1)
    
    # Create polyp region - elliptical shape
    center = (int(img_size[0]*0.6), int(img_size[1]*0.4))
    axes = (int(img_size[0]*0.15), int(img_size[1]*0.2))
    angle = 30
    
    # Create ground truth mask
    gt_mask = np.zeros(img_size, dtype=np.float32)
    cv2.ellipse(gt_mask, center, axes, angle, 0, 360, 1, thickness=-1)
    
    # Add some irregularity
    noise = np.random.normal(0, 1, gt_mask.shape) * 0.1
    gt_mask = np.clip(gt_mask + noise, 0, 1)
    gt_mask = (gt_mask > 0.5).astype(np.float32)
    
    # Make polyp region more visible in the image
    polyp_color = np.array([0.9, 0.6, 0.6])
    for c in range(3):
        background[:,:,c] = background[:,:,c] * (1 - gt_mask) + polyp_color[c] * gt_mask
    
    # Add some texture to the polyp
    texture = np.random.normal(0, 1, (img_size[0], img_size[1])) * 0.05
    for c in range(3):
        background[:,:,c] = np.clip(background[:,:,c] + texture * gt_mask, 0, 1)
    
    # Step 2: Create mask sequence showing RL agent's iterative improvements
    num_steps = 6  # Including initial state
    
    # Initialize sequences
    image_sequence = [background.copy() for _ in range(num_steps)]
    mask_sequence = [np.zeros(img_size, dtype=np.float32) for _ in range(num_steps)]
    pointer_sequence = []
    action_sequence = ["Initial State", "Move Pointer", "Expand Region", "Move Pointer", "Expand Region", "Confirm Segmentation"]
    
    # Initial pointer position - random location near polyp
    initial_pointer = (int(center[0] + np.random.normal(0, 10)), 
                      int(center[1] + np.random.normal(0, 10)))
    pointer_sequence.append(initial_pointer)
    
    # Step 1: Move pointer near polyp center
    pointer2 = (int(center[0] + np.random.normal(0, 5)), 
               int(center[1] + np.random.normal(0, 5)))
    pointer_sequence.append(pointer2)
    
    # Mask remains unchanged
    mask_sequence[1] = mask_sequence[0].copy()
    
    # Step 2: Expand region - create small circle as initial segmentation
    small_circle = np.zeros(img_size, dtype=np.float32)
    cv2.circle(small_circle, pointer2, int(min(axes)/3), 1, thickness=-1)
    mask_sequence[2] = small_circle
    pointer_sequence.append(pointer2)  # Pointer position remains unchanged
    
    # Step 3: Move pointer to polyp edge
    edge_pointer = (
        int(center[0] + axes[0]*0.8*np.cos(np.deg2rad(angle+30))),
        int(center[1] + axes[1]*0.8*np.sin(np.deg2rad(angle+30)))
    )
    pointer_sequence.append(edge_pointer)
    mask_sequence[3] = mask_sequence[2].copy()  # Mask remains unchanged
    
    # Step 4: Expand region - enlarge segmentation area
    mask_sequence[4] = np.zeros(img_size, dtype=np.float32)
    cv2.ellipse(mask_sequence[4], center, 
               (int(axes[0]*0.85), int(axes[1]*0.85)), 
               angle, 0, 360, 1, thickness=-1)
    pointer_sequence.append(edge_pointer)  # Pointer position remains unchanged
    
    # Step 5: Confirm segmentation - final result, close to ground truth but with some differences
    mask_sequence[5] = np.zeros(img_size, dtype=np.float32)
    cv2.ellipse(mask_sequence[5], center, 
               (int(axes[0]*0.95), int(axes[1]*0.95)), 
               angle, 0, 360, 1, thickness=-1)
    pointer_sequence.append(edge_pointer)  # Pointer position remains unchanged
    
    # Add some random noise to final mask to differentiate from ground truth
    noise = np.random.normal(0, 1, mask_sequence[5].shape) * 0.05
    mask_sequence[5] = np.clip(mask_sequence[5] + noise, 0, 1)
    mask_sequence[5] = (mask_sequence[5] > 0.5).astype(np.float32)
    
    return {
        'image': background,
        'gt_mask': gt_mask,
        'image_sequence': image_sequence,
        'mask_sequence': mask_sequence,
        'pointer_sequence': pointer_sequence,
        'action_sequence': action_sequence,
        'num_steps': num_steps
    }

def overlay_mask(img, mask, color=(1.0, 0.5, 0.0), alpha=0.35):
    """Overlay transparent mask onto an image"""
    # Ensure img and mask are 3-channel and 1-channel
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=2)
    
    if len(mask.shape) == 3 and mask.shape[2] > 1:
        mask = mask[:,:,0]  # Use only first channel
    
    # Create colored mask
    colored_mask = np.zeros_like(img)
    for i in range(3):
        colored_mask[:,:,i] = mask * color[i]
    
    # Create transparent overlay
    result = img.copy()
    idx = mask > 0
    for c in range(3):
        result[:,:,c][idx] = result[:,:,c][idx] * (1-alpha) + colored_mask[:,:,c][idx] * alpha
    
    return result

def draw_pointer(img, point, color=(0.0, 1.0, 0.0), radius=5):
    """Draw pointer marker on image"""
    result = img.copy()
    cv2.circle(result, point, radius, color, thickness=2)
    cv2.circle(result, point, 1, color, thickness=-1)
    
    # Draw crosshairs
    length = radius - 1
    cv2.line(result, 
             (point[0]-length, point[1]), 
             (point[0]+length, point[1]), 
             color, thickness=1)
    cv2.line(result, 
             (point[0], point[1]-length), 
             (point[0], point[1]+length), 
             color, thickness=1)
    
    return result

def add_text_to_image(img, text, position, font_size=12, color=(1, 1, 1), bg_color=None):
    """Add text label to image"""
    # Convert to PIL Image to use ImageDraw
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img_pil)
    
    # Load font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # If background color provided, draw background first
    if bg_color:
        # Estimate text dimensions
        text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
        draw.rectangle(
            [position[0], position[1], position[0] + text_width, position[1] + text_height],
            fill=bg_color
        )
    
    # Draw text
    draw.text(position, text, font=font, fill=tuple(int(c*255) for c in color))
    
    # Convert back to NumPy array
    return np.array(img_pil) / 255.0

def create_interactive_rl_visualization(data):
    """Create visualization of interactive RL process"""
    num_steps = data['num_steps']
    fig = plt.figure(figsize=(12, 9), dpi=300)
    
    # Create grid layout
    gs = GridSpec(3, num_steps, height_ratios=[1, 1, 1])
    
    # Top row: Original image and ground truth mask - centered in the layout
    ax_original = plt.subplot(gs[0, 1:3])
    ax_original.imshow(data['image'])
    ax_original.set_title('Original Colonoscopy Image')
    ax_original.axis('off')
    
    ax_gt = plt.subplot(gs[0, 3:5])
    ax_gt.imshow(overlay_mask(data['image'], data['gt_mask']))
    ax_gt.set_title('Ground Truth Segmentation (Expert)')
    ax_gt.axis('off')
    
    # Middle and bottom rows: Each step of the RL process
    for step in range(num_steps):
        # Middle row: Show current mask
        ax_step = plt.subplot(gs[1, step])
        
        # Overlay mask and pointer
        img_with_mask = overlay_mask(data['image_sequence'][step], 
                                    data['mask_sequence'][step])
        img_with_pointer = draw_pointer(img_with_mask, 
                                       data['pointer_sequence'][step])
        
        ax_step.imshow(img_with_pointer)
        ax_step.set_title(f'Step {step+1}')
        ax_step.axis('off')
        
        # Bottom row: Show comparison between current and ground truth masks
        ax_comp = plt.subplot(gs[2, step])
        
        # Create mask comparison image
        # Red: predicted mask, Blue: ground truth mask, Purple: overlap
        comparison = np.zeros((*data['gt_mask'].shape, 3), dtype=np.float32)
        comparison[:,:,0] = data['mask_sequence'][step]  # Red channel - prediction
        comparison[:,:,2] = data['gt_mask']  # Blue channel - ground truth
        
        ax_comp.imshow(comparison)
        
        # Calculate current Dice score
        intersection = np.logical_and(data['mask_sequence'][step] > 0.5, 
                                     data['gt_mask'] > 0.5).sum()
        dice = (2. * intersection) / (data['mask_sequence'][step].sum() + data['gt_mask'].sum() + 1e-6)
        
        ax_comp.set_title(f'Dice: {dice:.4f}')
        ax_comp.axis('off')
        
        # Add action labels
        if step < len(data['action_sequence']):
            action_text = data['action_sequence'][step]
            plt.figtext(0.1 + step * 0.8 / (num_steps-1), 0.52, 
                       action_text, ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round'))
    
    plt.suptitle('Interactive RL Segmentation Process Visualization', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.1, wspace=0.05)
    
    # Save images
    plt.savefig(os.path.join(output_dir, 'interactive_rl_process.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'interactive_rl_process.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def main():
    """Main function"""
    print("Generating InteractiveRL process visualization...")
    
    # Generate simulated data
    data = generate_synthetic_rl_process()
    
    # Create visualization
    create_interactive_rl_visualization(data)
    
    print(f"InteractiveRL process visualization saved to {output_dir} directory")

if __name__ == "__main__":
    main() 