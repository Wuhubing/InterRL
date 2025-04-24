import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.transform import resize
from PIL import Image, ImageDraw
import cv2
from scipy.ndimage import distance_transform_edt
import gym
from gym import spaces

class SegmentationAction:
    """Helper class for segmentation actions"""
    # Movement actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7
    # Segmentation actions
    EXPAND = 8
    SHRINK = 9

class EnhancedSegmentationEnv(gym.Env):
    """Enhanced environment for interactive polyp segmentation"""
    
    def __init__(self, 
                 images, masks, 
                 image_size=(256, 256),
                 max_steps=100,
                 step_penalty=-0.01,
                 dice_weight=0.8,
                 boundary_weight=0.2,
                 expansion_factor=1.05,
                 shrink_factor=0.95,
                 pointer_radius=10,
                 movement_step=10,
                 difficulty='easy'):
        """Initialize segmentation environment
        
        Args:
            images: List of images
            masks: List of ground truth masks
            image_size: Size to resize images to
            max_steps: Maximum number of steps per episode
            step_penalty: Penalty for each step
            dice_weight: Weight for dice coefficient in reward
            boundary_weight: Weight for boundary accuracy in reward
            expansion_factor: Factor to expand mask by
            shrink_factor: Factor to shrink mask by
            pointer_radius: Radius of pointer
            movement_step: Step size for pointer movement
            difficulty: Difficulty level ('easy', 'medium', 'hard')
        """
        super(EnhancedSegmentationEnv, self).__init__()
        
        # Store parameters
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.expansion_factor = expansion_factor
        self.shrink_factor = shrink_factor
        self.pointer_radius = pointer_radius
        self.base_pointer_radius = pointer_radius  # Store original radius for difficulty adjustment
        self.movement_step = movement_step
        self.difficulty = difficulty
        
        # Adjust difficulty
        self._adjust_difficulty(difficulty)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(10)  # 8 movement, expand, shrink
        
        # Observation space: [image, current_mask, pointer, distance_map, edge_map]
        # 3 channels for image, 1 for mask, 1 for pointer, 1 for distance map, 1 for edge map
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(7, *image_size),  # 7 channels 
            dtype=np.float32
        )
        
        # Initialize state
        self.current_image_idx = 0
        self.image = None
        self.gt_mask = None
        self.current_mask = None
        self.pointer_x = 0
        self.pointer_y = 0
        self.steps = 0
        self.previous_dice = 0
        self.previous_iou = 0
        
        # History for curriculum learning
        self.episode_history = {
            'dice_scores': [],
            'iou_scores': [],
            'rewards': [],
            'steps': []
        }
        
        # Initialize visualization parameters
        self.visualization_enabled = False
        self.vis_history = []
    
    def _adjust_difficulty(self, difficulty):
        """Adjust environment parameters based on difficulty level
        
        Args:
            difficulty: Difficulty level ('easy', 'medium', 'hard')
        """
        if difficulty == 'easy':
            # Start with a larger pointer and more forgiving penalties
            self.pointer_radius = self.base_pointer_radius * 1.5
            self.step_penalty = -0.005
            self.expansion_factor = 1.1
            self.shrink_factor = 0.9
        elif difficulty == 'medium':
            # Default values
            self.pointer_radius = self.base_pointer_radius
            self.step_penalty = -0.01
            self.expansion_factor = 1.05
            self.shrink_factor = 0.95
        elif difficulty == 'hard':
            # Smaller pointer, harsher penalties, more precise actions
            self.pointer_radius = self.base_pointer_radius * 0.8
            self.step_penalty = -0.02
            self.expansion_factor = 1.02
            self.shrink_factor = 0.98
            self.movement_step = max(5, self.movement_step // 2)  # Smaller movement steps
    
    def set_difficulty(self, difficulty):
        """Set difficulty level
        
        Args:
            difficulty: Difficulty level ('easy', 'medium', 'hard')
        """
        self.difficulty = difficulty
        self._adjust_difficulty(difficulty)
    
    def enable_visualization(self, enabled=True):
        """Enable or disable visualization
        
        Args:
            enabled: Whether to enable visualization
        """
        self.visualization_enabled = enabled
    
    def _preprocess_image_and_mask(self, image, mask):
        """Preprocess image and mask
        
        Args:
            image: Image to preprocess
            mask: Mask to preprocess
        
        Returns:
            Preprocessed image and mask
        """
        # Resize image and mask to desired size
        image = resize(image, self.image_size, preserve_range=True).astype(np.float32)
        mask = resize(mask, self.image_size, preserve_range=True).astype(np.float32)
        
        # Normalize image to [0, 1]
        image = image / 255.0
        
        # Ensure mask is binary
        mask = (mask > 0.5).astype(np.float32)
        
        return image, mask
    
    def _create_distance_map(self, mask):
        """Create distance map from mask
        
        Args:
            mask: Binary mask
        
        Returns:
            Distance map
        """
        # Create distance transform
        dist_transform = distance_transform_edt(mask)
        
        # Normalize to [0, 1]
        if dist_transform.max() > 0:
            dist_transform = dist_transform / dist_transform.max()
        
        return dist_transform
    
    def _create_edge_map(self, mask):
        """Create edge map from mask
        
        Args:
            mask: Binary mask
        
        Returns:
            Edge map
        """
        # Create edge map using Canny edge detector
        mask_uint8 = (mask * 255).astype(np.uint8)
        edges = cv2.Canny(mask_uint8, 100, 200) / 255.0
        
        return edges.astype(np.float32)
    
    def _create_pointer_mask(self, center_x, center_y, radius, image_shape):
        """Create circular pointer mask
        
        Args:
            center_x: X coordinate of pointer center
            center_y: Y coordinate of pointer center
            radius: Pointer radius
            image_shape: Shape of image
        
        Returns:
            Pointer mask
        """
        # Create empty mask
        mask = np.zeros(image_shape, dtype=np.float32)
        
        # Create circular mask
        y, x = np.ogrid[:image_shape[0], :image_shape[1]]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create soft circular mask with Gaussian falloff
        sigma = radius / 3.0
        mask = np.exp(-(dist_from_center**2) / (2 * sigma**2))
        
        # Normalize to [0, 1]
        mask = mask / mask.max()
        
        return mask
    
    def _generate_initial_mask(self, gt_mask):
        """Generate initial mask by selecting a point within the ground truth
        
        Args:
            gt_mask: Ground truth mask
        
        Returns:
            Initial mask and pointer coordinates
        """
        # Find non-zero coordinates in ground truth mask
        y_coords, x_coords = np.where(gt_mask > 0.5)
        
        if len(y_coords) == 0:
            # If no foreground pixels, place pointer in the center
            center_y, center_x = self.image_size[0] // 2, self.image_size[1] // 2
        else:
            # Randomly select a foreground point
            idx = np.random.randint(0, len(y_coords))
            center_y, center_x = y_coords[idx], x_coords[idx]
        
        # Create initial mask
        init_mask = self._create_pointer_mask(center_x, center_y, self.pointer_radius, gt_mask.shape)
        
        return init_mask, center_x, center_y
    
    def _compute_metrics(self, pred_mask, gt_mask, eps=1e-6):
        """Compute segmentation metrics
        
        Args:
            pred_mask: Predicted mask
            gt_mask: Ground truth mask
            eps: Small value to avoid division by zero
        
        Returns:
            Dictionary of metrics
        """
        # Binarize masks
        pred_bin = (pred_mask > 0.5).astype(np.float32)
        gt_bin = (gt_mask > 0.5).astype(np.float32)
        
        # Compute intersection and union
        intersection = np.sum(pred_bin * gt_bin)
        union = np.sum(pred_bin) + np.sum(gt_bin) - intersection
        
        # Compute Dice coefficient and IoU
        dice = (2 * intersection + eps) / (np.sum(pred_bin) + np.sum(gt_bin) + eps)
        iou = (intersection + eps) / (union + eps)
        
        # Compute boundary F1 score
        pred_edges = self._create_edge_map(pred_bin)
        gt_edges = self._create_edge_map(gt_bin)
        
        # Dilate edges slightly for more forgiving boundary matching
        kernel = np.ones((3, 3), np.uint8)
        pred_edges = cv2.dilate((pred_edges * 255).astype(np.uint8), kernel, iterations=1) / 255.0
        gt_edges = cv2.dilate((gt_edges * 255).astype(np.uint8), kernel, iterations=1) / 255.0
        
        # Compute precision and recall for boundaries
        tp = np.sum(pred_edges * gt_edges)
        fp = np.sum(pred_edges) - tp
        fn = np.sum(gt_edges) - tp
        
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        
        boundary_f1 = (2 * precision * recall) / (precision + recall + eps)
        
        return {
            'dice': dice,
            'iou': iou,
            'boundary_f1': boundary_f1
        }
    
    def _compute_reward(self, metrics, previous_metrics=None):
        """Compute reward based on metrics
        
        Args:
            metrics: Current metrics
            previous_metrics: Previous metrics
        
        Returns:
            Reward
        """
        # Extract metrics
        dice = metrics['dice']
        boundary_f1 = metrics['boundary_f1']
        
        # Compute weighted reward
        reward = self.dice_weight * dice + self.boundary_weight * boundary_f1
        
        # Add step penalty
        reward += self.step_penalty
        
        # Add improvement bonus if previous metrics are available
        if previous_metrics is not None:
            dice_improvement = dice - previous_metrics['dice']
            
            # Give bonus for improvements
            if dice_improvement > 0:
                reward += dice_improvement * 0.5
        
        return reward
    
    def _update_mask(self, action):
        """Update mask based on action
        
        Args:
            action: Action index
        
        Returns:
            Updated mask
        """
        if action == SegmentationAction.EXPAND:
            # Expand current mask
            if self.expansion_factor > 1.0:
                # Apply morphological dilation
                kernel_size = int(max(2, self.pointer_radius // 4))
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask_uint8 = (self.current_mask * 255).astype(np.uint8)
                expanded = cv2.dilate(mask_uint8, kernel, iterations=1) / 255.0
                
                # Blend with original
                alpha = (self.expansion_factor - 1.0) / 0.1  # Map [1.0, 1.1] to [0, 1]
                alpha = min(1.0, max(0.0, alpha))
                return alpha * expanded + (1 - alpha) * self.current_mask
            return self.current_mask
            
        elif action == SegmentationAction.SHRINK:
            # Shrink current mask
            if self.shrink_factor < 1.0:
                # Apply morphological erosion
                kernel_size = int(max(2, self.pointer_radius // 4))
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask_uint8 = (self.current_mask * 255).astype(np.uint8)
                shrunk = cv2.erode(mask_uint8, kernel, iterations=1) / 255.0
                
                # Blend with original
                alpha = (1.0 - self.shrink_factor) / 0.1  # Map [0.9, 1.0] to [1, 0]
                alpha = min(1.0, max(0.0, alpha))
                return alpha * shrunk + (1 - alpha) * self.current_mask
            return self.current_mask
            
        else:
            # Movement action, update pointer position
            if action == SegmentationAction.UP:
                self.pointer_y = max(0, self.pointer_y - self.movement_step)
            elif action == SegmentationAction.DOWN:
                self.pointer_y = min(self.image_size[0] - 1, self.pointer_y + self.movement_step)
            elif action == SegmentationAction.LEFT:
                self.pointer_x = max(0, self.pointer_x - self.movement_step)
            elif action == SegmentationAction.RIGHT:
                self.pointer_x = min(self.image_size[1] - 1, self.pointer_x + self.movement_step)
            elif action == SegmentationAction.UP_LEFT:
                self.pointer_y = max(0, self.pointer_y - self.movement_step)
                self.pointer_x = max(0, self.pointer_x - self.movement_step)
            elif action == SegmentationAction.UP_RIGHT:
                self.pointer_y = max(0, self.pointer_y - self.movement_step)
                self.pointer_x = min(self.image_size[1] - 1, self.pointer_x + self.movement_step)
            elif action == SegmentationAction.DOWN_LEFT:
                self.pointer_y = min(self.image_size[0] - 1, self.pointer_y + self.movement_step)
                self.pointer_x = max(0, self.pointer_x - self.movement_step)
            elif action == SegmentationAction.DOWN_RIGHT:
                self.pointer_y = min(self.image_size[0] - 1, self.pointer_y + self.movement_step)
                self.pointer_x = min(self.image_size[1] - 1, self.pointer_x + self.movement_step)
            
            # Create new pointer mask and integrate with current mask
            pointer_mask = self._create_pointer_mask(self.pointer_x, self.pointer_y, self.pointer_radius, self.image_size)
            
            # Blend with current mask using maximum
            return np.maximum(self.current_mask, pointer_mask)
    
    def _get_state(self):
        """Get current state observation
        
        Returns:
            State observation
        """
        # Create pointer mask
        pointer_mask = self._create_pointer_mask(self.pointer_x, self.pointer_y, self.pointer_radius, self.image_size)
        
        # Create distance map
        distance_map = self._create_distance_map(self.current_mask)
        
        # Create edge map
        edge_map = self._create_edge_map(self.current_mask)
        
        # Stack channels
        state = np.stack([
            self.image[:, :, 0],   # R channel
            self.image[:, :, 1],   # G channel
            self.image[:, :, 2],   # B channel
            self.current_mask,     # Current mask
            pointer_mask,          # Pointer position
            distance_map,          # Distance map
            edge_map               # Edge map
        ], axis=0)
        
        return state.astype(np.float32)
    
    def _get_info(self):
        """Get additional information about current state
        
        Returns:
            Dictionary of information
        """
        # Compute metrics
        metrics = self._compute_metrics(self.current_mask, self.gt_mask)
        
        return {
            'dice': metrics['dice'],
            'iou': metrics['iou'],
            'boundary_f1': metrics['boundary_f1'],
            'steps': self.steps,
            'pointer_x': self.pointer_x,
            'pointer_y': self.pointer_y,
            'difficulty': self.difficulty
        }
    
    def reset(self, image_idx=None):
        """Reset environment
        
        Args:
            image_idx: Index of image to use, if None a random image is selected
        
        Returns:
            Initial state observation
        """
        # Select image
        if image_idx is None:
            self.current_image_idx = np.random.randint(0, len(self.images))
        else:
            self.current_image_idx = image_idx % len(self.images)
        
        # Load image and mask
        self.image, self.gt_mask = self._preprocess_image_and_mask(
            self.images[self.current_image_idx],
            self.masks[self.current_image_idx]
        )
        
        # Generate initial mask
        self.current_mask, self.pointer_x, self.pointer_y = self._generate_initial_mask(self.gt_mask)
        
        # Reset step counter
        self.steps = 0
        
        # Reset previous metrics
        metrics = self._compute_metrics(self.current_mask, self.gt_mask)
        self.previous_dice = metrics['dice']
        self.previous_iou = metrics['iou']
        
        # Reset visualization history
        self.vis_history = []
        if self.visualization_enabled:
            self._add_to_vis_history()
        
        # Get observation
        obs = self._get_state()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """Take step in environment
        
        Args:
            action: Action index
        
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Update mask based on action
        self.current_mask = self._update_mask(action)
        
        # Increment step counter
        self.steps += 1
        
        # Compute metrics
        metrics = self._compute_metrics(self.current_mask, self.gt_mask)
        
        # Compute reward
        previous_metrics = {
            'dice': self.previous_dice,
            'iou': self.previous_iou
        }
        reward = self._compute_reward(metrics, previous_metrics)
        
        # Update previous metrics
        self.previous_dice = metrics['dice']
        self.previous_iou = metrics['iou']
        
        # Check if episode is done
        done = (self.steps >= self.max_steps)
        
        # Add to visualization history
        if self.visualization_enabled:
            self._add_to_vis_history()
        
        # Get observation and info
        obs = self._get_state()
        info = self._get_info()
        
        # If episode is done, update episode history
        if done:
            self.episode_history['dice_scores'].append(metrics['dice'])
            self.episode_history['iou_scores'].append(metrics['iou'])
            self.episode_history['rewards'].append(reward)
            self.episode_history['steps'].append(self.steps)
        
        return obs, reward, done, info
    
    def _add_to_vis_history(self):
        """Add current state to visualization history"""
        # Create visualization
        vis_image = self.image.copy()
        
        # Add mask overlay
        mask_overlay = mark_boundaries(vis_image, (self.current_mask > 0.5).astype(np.int32), color=(1, 0, 0), mode='thick')
        
        # Add ground truth boundary
        gt_overlay = mark_boundaries(mask_overlay, (self.gt_mask > 0.5).astype(np.int32), color=(0, 1, 0), mode='thick')
        
        # Add pointer
        pointer_img = Image.fromarray((gt_overlay * 255).astype(np.uint8))
        draw = ImageDraw.Draw(pointer_img)
        draw.ellipse(
            (self.pointer_x - self.pointer_radius, self.pointer_y - self.pointer_radius,
             self.pointer_x + self.pointer_radius, self.pointer_y + self.pointer_radius),
            outline='blue'
        )
        gt_overlay = np.array(pointer_img) / 255.0
        
        # Add current metrics
        metrics = self._compute_metrics(self.current_mask, self.gt_mask)
        info = {
            'step': self.steps,
            'dice': metrics['dice'],
            'iou': metrics['iou'],
            'boundary_f1': metrics['boundary_f1']
        }
        
        # Add to history
        self.vis_history.append((gt_overlay, info))
    
    def render_history(self, save_path=None, figsize=(15, 10), ncols=5):
        """Render visualization history
        
        Args:
            save_path: Path to save visualization
            figsize: Figure size
            ncols: Number of columns in grid
        """
        if not self.visualization_enabled or len(self.vis_history) == 0:
            print("Visualization is not enabled or history is empty")
            return
        
        # Calculate layout
        num_frames = len(self.vis_history)
        nrows = (num_frames + ncols - 1) // ncols
        
        # Create figure
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
        
        # Plot frames
        for i, (frame, info) in enumerate(self.vis_history):
            if i < len(axes):
                axes[i].imshow(frame)
                axes[i].set_title(f"Step {info['step']} - Dice: {info['dice']:.3f}")
                axes[i].axis('off')
        
        # Hide empty subplots
        for i in range(num_frames, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def get_episode_stats(self):
        """Get statistics from completed episodes
        
        Returns:
            Dictionary of episode statistics
        """
        if len(self.episode_history['dice_scores']) == 0:
            return {
                'avg_dice': 0,
                'avg_iou': 0,
                'avg_reward': 0,
                'avg_steps': 0
            }
        
        return {
            'avg_dice': np.mean(self.episode_history['dice_scores']),
            'avg_iou': np.mean(self.episode_history['iou_scores']),
            'avg_reward': np.mean(self.episode_history['rewards']),
            'avg_steps': np.mean(self.episode_history['steps'])
        }
    
    def clear_episode_history(self):
        """Clear episode history"""
        for key in self.episode_history:
            self.episode_history[key] = []

def create_segmentation_env(images, masks, image_size=(256, 256), difficulty='medium'):
    """Create segmentation environment
    
    Args:
        images: List of images
        masks: List of ground truth masks
        image_size: Size to resize images to
        difficulty: Difficulty level ('easy', 'medium', 'hard')
    
    Returns:
        Segmentation environment
    """
    env = EnhancedSegmentationEnv(
        images=images,
        masks=masks,
        image_size=image_size,
        difficulty=difficulty
    )
    
    return env 