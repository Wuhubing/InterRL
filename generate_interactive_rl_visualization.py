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

# Set matplotlib parameters for academic-style charts - 与academic_plots保持一致
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

# 设置与academic_plots一致的坐标轴样式
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['text.color'] = 'black'
plt.rcParams['grid.color'] = 'lightgray'
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3

# 设置新的配色方案 - 科研画图风格
COLOR_PALETTE = ['#23BAC5', '#EECA40', '#FD763F']
COLOR_UNET = COLOR_PALETTE[0]      # 蓝绿色
COLOR_RL = COLOR_PALETTE[1]        # 金黄色
COLOR_TEST = COLOR_PALETTE[2]      # 橙红色
COLOR_VAL = '#F8F3EB'             # 保留浅米色作为背景色

# 定义交互可视化中使用的颜色
COLOR_ORIGINAL = '#777777'        # 灰色 - 原始图像
COLOR_MASK = COLOR_RL             # 使用RL的金黄色 - 分割掩码
COLOR_POINTER = COLOR_TEST        # 使用测试的橙红色 - 指针位置
COLOR_ACTION = COLOR_UNET         # 使用UNet的蓝绿色 - 动作
COLOR_GT = COLOR_UNET             # 改为蓝绿色 - 真值掩码
COLOR_OVERLAP = '#9966CC'         # 紫色 - 重叠区域

# Create output directory - 与academic_plots一致
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

def overlay_mask(img, mask, color_tuple=None, alpha=0.35):
    """Overlay transparent mask onto an image"""
    if color_tuple is None:
        # 默认使用RL金黄色
        color = tuple(int(c*255) for c in mpl.colors.to_rgb(COLOR_MASK))
    else:
        color = color_tuple
        
    # 确保RGB值在0-1范围内
    if isinstance(color[0], float) and color[0] <= 1.0:
        color = tuple(int(c*255) for c in color)
    
    # Ensure img and mask are 3-channel and 1-channel
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=2)
    
    if len(mask.shape) == 3 and mask.shape[2] > 1:
        mask = mask[:,:,0]  # Use only first channel
    
    # Convert to uint8 for better visualization
    img_uint8 = (img * 255).astype(np.uint8)
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Create mask overlay
    result = img_uint8.copy()
    mask_rgb = np.zeros_like(result)
    mask_rgb[mask_uint8 > 127] = color
    
    # Blend the images
    cv2.addWeighted(mask_rgb, alpha, result, 1 - alpha, 0, result)
    
    return result / 255.0

def draw_pointer(img, point, color=None, radius=5):
    """Draw pointer marker on image"""
    if color is None:
        # 默认使用测试橙红色
        color = tuple(int(c*255) for c in mpl.colors.to_rgb(COLOR_POINTER))
    
    # 确保RGB值在0-255范围
    if isinstance(color[0], float) and color[0] <= 1.0:
        color = tuple(int(c*255) for c in color)
    
    result = img.copy()
    if isinstance(result[0,0,0], np.float32):
        result = (result * 255).astype(np.uint8)
    
    # Draw crosshair pointer with academic style
    cv2.circle(result, point, radius, color, thickness=2)
    
    # Draw crosshairs
    length = radius * 2
    cv2.line(result, 
             (point[0]-length, point[1]), 
             (point[0]+length, point[1]), 
             color, thickness=1)
    cv2.line(result, 
             (point[0], point[1]-length), 
             (point[0], point[1]+length), 
             color, thickness=1)
    
    return result / 255.0 if result.dtype == np.uint8 else result

def add_text_to_image(img, text, position, font_size=12, color=None, bg_color=None):
    """Add text label to image with academic style"""
    if color is None:
        # 默认使用黑色文本
        color = (0, 0, 0)
    
    # Convert to PIL Image to use ImageDraw
    img_np = img
    if isinstance(img[0,0,0], np.float32):
        img_np = (img * 255).astype(np.uint8)
    
    img_pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img_pil)
    
    # Load font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # If background color provided, draw background first
    if bg_color:
        # Estimate text dimensions
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2]-text_bbox[0], text_bbox[3]-text_bbox[1]
        padding = 2
        
        draw.rectangle(
            [position[0]-padding, position[1]-padding, 
             position[0] + text_width + padding, position[1] + text_height + padding],
            fill=bg_color
        )
    
    # Draw text
    draw.text(position, text, font=font, fill=color)
    
    # Convert back to NumPy array
    return np.array(img_pil) / 255.0

def create_interactive_rl_visualization(data):
    """Create visualization of interactive RL process with academic style"""
    num_steps = data['num_steps']
    
    # Use a clean style consistent with academic plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with academic proportions
    fig = plt.figure(figsize=(14, 10), dpi=300)
    
    # Create grid layout with proper spacing
    gs = GridSpec(3, num_steps, height_ratios=[1, 1.2, 1], hspace=0.3, wspace=0.1)
    
    # Top row: Original image and ground truth mask - centered in the layout
    ax_original = plt.subplot(gs[0, 1:3])
    ax_original.imshow(data['image'])
    ax_original.set_title('Original Colonoscopy Image', fontweight='bold')
    ax_original.axis('off')
    
    ax_gt = plt.subplot(gs[0, 3:5])
    # 使用科研配色方案覆盖掩码
    gt_color = tuple(c for c in mpl.colors.to_rgb(COLOR_GT))
    ax_gt.imshow(overlay_mask(data['image'], data['gt_mask'], color_tuple=gt_color))
    ax_gt.set_title('Ground Truth Segmentation', fontweight='bold')
    ax_gt.axis('off')
    
    # 计算全局最佳Dice评分，用于标记
    best_dice = 0
    for step in range(num_steps):
        intersection = np.logical_and(data['mask_sequence'][step] > 0.5, 
                                     data['gt_mask'] > 0.5).sum()
        dice = (2. * intersection) / (data['mask_sequence'][step].sum() + data['gt_mask'].sum() + 1e-6)
        best_dice = max(best_dice, dice)
    
    # Middle and bottom rows: Each step of the RL process
    for step in range(num_steps):
        # Middle row: Show current mask
        ax_step = plt.subplot(gs[1, step])
        
        # 使用科研风格颜色方案
        mask_color = tuple(c for c in mpl.colors.to_rgb(COLOR_MASK))
        pointer_color = tuple(c for c in mpl.colors.to_rgb(COLOR_POINTER))
        
        # Overlay mask and pointer
        img_with_mask = overlay_mask(data['image_sequence'][step], 
                                   data['mask_sequence'][step],
                                   color_tuple=mask_color)
        img_with_pointer = draw_pointer(img_with_mask, 
                                      data['pointer_sequence'][step],
                                      color=pointer_color)
        
        ax_step.imshow(img_with_pointer)
        ax_step.set_title(f'Step {step+1}', fontweight='bold')
        ax_step.axis('off')
        
        # Bottom row: Show comparison between current and ground truth masks
        ax_comp = plt.subplot(gs[2, step])
        
        # Create mask comparison image with academic colors
        # Yellow/Gold: predicted mask (RL), Blue/Teal: ground truth mask (UNet), Purple: overlap
        comparison = np.zeros((*data['gt_mask'].shape, 3), dtype=np.float32)
        
        # Convert hex colors to RGB for the comparison visualization
        mask_rgb = np.array(mpl.colors.to_rgb(COLOR_MASK))  # RL gold
        gt_rgb = np.array(mpl.colors.to_rgb(COLOR_GT))      # UNet teal
        overlap_rgb = np.array(mpl.colors.to_rgb(COLOR_OVERLAP))  # Purple
        
        # 创建比较图像
        gt_mask_bool = data['gt_mask'] > 0.5
        pred_mask_bool = data['mask_sequence'][step] > 0.5
        overlap_mask = np.logical_and(gt_mask_bool, pred_mask_bool)
        
        # 设置每个区域的颜色
        for c in range(3):
            comparison[:,:,c] = np.where(np.logical_and(pred_mask_bool, ~overlap_mask), 
                                         mask_rgb[c], 0)
            comparison[:,:,c] += np.where(np.logical_and(gt_mask_bool, ~overlap_mask), 
                                          gt_rgb[c], 0)
            comparison[:,:,c] += np.where(overlap_mask, overlap_rgb[c], 0)
        
        ax_comp.imshow(comparison)
        
        # Calculate current Dice score
        intersection = np.logical_and(data['mask_sequence'][step] > 0.5, 
                                    data['gt_mask'] > 0.5).sum()
        dice = (2. * intersection) / (data['mask_sequence'][step].sum() + data['gt_mask'].sum() + 1e-6)
        
        # 使用学术风格设置标题
        if dice == best_dice and step > 0:
            ax_comp.set_title(f'Dice: {dice:.4f} (Best)', fontweight='bold')
        else:
            ax_comp.set_title(f'Dice: {dice:.4f}', fontweight='bold')
        ax_comp.axis('off')
        
        # Add action labels with academic style
        if step < len(data['action_sequence']):
            action_text = data['action_sequence'][step]
            # 使用与学术图表一致的文本框样式
            plt.figtext(
                0.1 + step * 0.8 / (num_steps-1), 0.52, 
                action_text, 
                ha='center', va='center',
                bbox=dict(
                    facecolor=COLOR_VAL, 
                    alpha=0.95, 
                    edgecolor='lightgray',
                    boxstyle='round,pad=0.5',
                    linewidth=0.8
                ),
                fontsize=10,
                color='black'
            )
    
    # 添加图例来解释颜色
    fig.legend(
        handles=[
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_MASK, markersize=10, label='Prediction'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_GT, markersize=10, label='Ground Truth'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_OVERLAP, markersize=10, label='Overlap'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_POINTER, markersize=10, label='Pointer')
        ],
        loc='upper right',
        bbox_to_anchor=(0.99, 0.99),
        frameon=True,
        facecolor=COLOR_VAL,
        edgecolor='lightgray'
    )
    
    plt.suptitle('Interactive Reinforcement Learning Segmentation Process', 
                fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.1)
    
    # Save images in high resolution
    plt.savefig(os.path.join(output_dir, 'interactive_rl_process.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'interactive_rl_process.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    
    # 打印最终Dice得分
    print(f"最终分割Dice分数: {dice:.4f}")
    print(f"最佳分割Dice分数: {best_dice:.4f}")
    print(f"可视化已保存到: {os.path.join(output_dir, 'interactive_rl_process.png')}")

def main():
    """Main function"""
    print("生成InteractiveRL过程可视化...")
    
    # 设置随机种子以确保结果一致
    np.random.seed(42)
    
    # Generate simulated data
    data = generate_synthetic_rl_process()
    
    # Create visualization with academic style
    create_interactive_rl_visualization(data)
    
    print(f"InteractiveRL过程可视化已保存至 {output_dir} 目录")

if __name__ == "__main__":
    main() 