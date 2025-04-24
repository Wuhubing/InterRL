import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.draw import disk
import gym
from gym import spaces
import torch
import os
import sys

# Add code directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.utils import dice_coefficient, iou_score

class EnhancedPolypSegmentationEnv(gym.Env):
    """
    Enhanced environment for interactive polyp segmentation using RL
    """
    
    def __init__(self, image, ground_truth, max_steps=100, pointer_radius=5,
                 expansion_factor=1.05, shrink_factor=0.95, move_step=5,
                 reward_weights=None, device='cpu', dense_rewards=True):
        """
        Initialize the environment
        
        Args:
            image (numpy.ndarray): RGB image
            ground_truth (numpy.ndarray): Binary ground truth mask
            max_steps (int): Maximum number of steps per episode
            pointer_radius (int): Radius of the pointer
            expansion_factor (float): Factor for expansion action
            shrink_factor (float): Factor for shrink action
            move_step (int): Step size for movement actions
            reward_weights (dict): Weights for different reward components
            device (str): Device to use for computations
            dense_rewards (bool): Whether to use dense rewards
        """
        # Store parameters
        self.image = image
        self.ground_truth = ground_truth.astype(np.float32)
        self.max_steps = max_steps
        self.pointer_radius = pointer_radius
        self.expansion_factor = expansion_factor
        self.shrink_factor = shrink_factor
        self.move_step = move_step
        self.device = device
        self.dense_rewards = dense_rewards
        
        # Set default reward weights if not provided
        self.reward_weights = {
            'dice': 1.0,
            'boundary': 0.3,
            'step_penalty': -0.01,
            'invalid_penalty': -0.05,
            'completion_bonus': 2.0,
            'exploration': 0.1
        }
        
        if reward_weights is not None:
            self.reward_weights.update(reward_weights)
        
        # Get image dimensions
        self.height, self.width = self.image.shape[:2]
        
        # Define action space (Up, Down, Left, Right, Expand, Shrink, Confirm)
        self.action_space = spaces.Discrete(7)
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=self.image.shape, dtype=np.uint8),
            'mask': spaces.Box(low=0, high=1, shape=(self.height, self.width), dtype=np.float32),
            'pointer_x': spaces.Box(low=0, high=self.width-1, shape=(1,), dtype=np.float32),
            'pointer_y': spaces.Box(low=0, high=self.height-1, shape=(1,), dtype=np.float32),
            'step_count': spaces.Box(low=0, high=self.max_steps, shape=(1,), dtype=np.float32),
            'visited_map': spaces.Box(low=0, high=1, shape=(self.height, self.width), dtype=np.float32),
        })
        
        # Initialize state variables
        self.pointer_x = None
        self.pointer_y = None
        self.mask = None
        self.step_count = None
        self.visited_map = None
        self.initial_centroid = None
        self.action_history = None
        self.cumulative_reward = None
        self.best_dice = None
        self.best_mask = None
        
        # Reset environment
        self.reset()
    
    def reset(self):
        """
        Reset the environment
        
        Returns:
            dict: Initial observation
        """
        # Reset step count
        self.step_count = 0
        
        # Reset action history
        self.action_history = []
        
        # Reset cumulative reward
        self.cumulative_reward = 0.0
        
        # Find centroid of the ground truth as the initial pointer position
        if self.ground_truth.sum() > 0:
            y_indices, x_indices = np.where(self.ground_truth > 0)
            self.initial_centroid = (int(np.mean(x_indices)), int(np.mean(y_indices)))
            self.pointer_x = np.array([self.initial_centroid[0]], dtype=np.float32)
            self.pointer_y = np.array([self.initial_centroid[1]], dtype=np.float32)
        else:
            # Fallback if the ground truth is empty
            self.initial_centroid = (self.width // 2, self.height // 2)
            self.pointer_x = np.array([self.initial_centroid[0]], dtype=np.float32)
            self.pointer_y = np.array([self.initial_centroid[1]], dtype=np.float32)
        
        # Initialize empty mask
        self.mask = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Initialize with a small circle at the pointer position
        rr, cc = disk(self.pointer_y[0], self.pointer_x[0], self.pointer_radius, shape=(self.height, self.width))
        self.mask[rr, cc] = 1.0
        
        # Keep track of the best mask (highest Dice coefficient)
        self.best_dice = self._calculate_dice()
        self.best_mask = self.mask.copy()
        
        # Initialize visited locations map
        self.visited_map = np.zeros((self.height, self.width), dtype=np.float32)
        self._update_visited_map()
        
        # Return initial observation
        return self._get_observation()
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action (int): Action to take (0: Up, 1: Down, 2: Left, 3: Right, 4: Expand, 5: Shrink, 6: Confirm)
            
        Returns:
            tuple: Observation, reward, done, info
        """
        # Store action
        self.action_history.append(action)
        
        # Update step count
        self.step_count += 1
        
        # Initialize reward and done flag
        reward = 0.0
        done = False
        
        # Initialize action validity flag
        valid_action = True
        
        # Get current Dice coefficient (before action)
        prev_dice = self._calculate_dice()
        
        # Process action
        if action == 0:  # Up
            self.pointer_y = np.maximum(self.pointer_y - self.move_step, 0)
            self._update_visited_map()
        elif action == 1:  # Down
            self.pointer_y = np.minimum(self.pointer_y + self.move_step, self.height - 1)
            self._update_visited_map()
        elif action == 2:  # Left
            self.pointer_x = np.maximum(self.pointer_x - self.move_step, 0)
            self._update_visited_map()
        elif action == 3:  # Right
            self.pointer_x = np.minimum(self.pointer_x + self.move_step, self.width - 1)
            self._update_visited_map()
        elif action == 4:  # Expand region
            new_mask = self._expand_region()
            if np.array_equal(new_mask, self.mask):
                valid_action = False
            else:
                self.mask = new_mask
        elif action == 5:  # Shrink region
            new_mask = self._shrink_region()
            if np.array_equal(new_mask, self.mask):
                valid_action = False
            else:
                self.mask = new_mask
        elif action == 6:  # Confirm segmentation
            # Episode is done when the confirm action is selected
            done = True
        
        # Calculate new Dice coefficient and reward
        new_dice = self._calculate_dice()
        reward = self._calculate_reward(prev_dice, new_dice, action, valid_action)
        
        # Update best mask if current is better
        if new_dice > self.best_dice:
            self.best_dice = new_dice
            self.best_mask = self.mask.copy()
        
        # Update cumulative reward
        self.cumulative_reward += reward
        
        # Check if maximum steps reached
        if self.step_count >= self.max_steps:
            done = True
        
        # If episode is done, use the best mask found
        if done and self.best_dice > new_dice:
            self.mask = self.best_mask
            new_dice = self.best_dice
        
        # Prepare info dictionary
        info = {
            'dice': new_dice,
            'iou': iou_score(self.mask, self.ground_truth),
            'steps': self.step_count,
            'cumulative_reward': self.cumulative_reward,
            'best_dice': self.best_dice,
            'valid_action': valid_action
        }
        
        # Return observation, reward, done flag, and info
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """
        Get the current observation
        
        Returns:
            dict: Observation
        """
        return {
            'image': self.image,
            'mask': self.mask,
            'pointer_x': self.pointer_x,
            'pointer_y': self.pointer_y,
            'step_count': np.array([self.step_count], dtype=np.float32),
            'visited_map': self.visited_map
        }
    
    def _update_visited_map(self):
        """
        Update the visited locations map
        """
        # Add current pointer position to visited map with Gaussian distribution
        y, x = int(self.pointer_y[0]), int(self.pointer_x[0])
        radius = max(3, self.pointer_radius // 2)
        
        rr, cc = disk(y, x, radius, shape=(self.height, self.width))
        self.visited_map[rr, cc] = 1.0
    
    def _expand_region(self):
        """
        Expand the current segmentation region
        
        Returns:
            numpy.ndarray: Updated mask
        """
        # Create a copy of the current mask
        new_mask = self.mask.copy()
        
        # Get binary mask and perform morphological dilation
        binary_mask = (new_mask > 0.5).astype(np.uint8)
        kernel_size = max(2, int(self.pointer_radius * self.expansion_factor))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        
        # Update the mask based on dilation, but only near the pointer
        # Create a distance-based weight centered at the pointer
        y, x = int(self.pointer_y[0]), int(self.pointer_x[0])
        y_grid, x_grid = np.ogrid[:self.height, :self.width]
        distance = np.sqrt((y_grid - y)**2 + (x_grid - x)**2)
        influence_radius = self.pointer_radius * 3
        weight = np.exp(-0.5 * (distance / influence_radius)**2)
        
        # Apply weighted expansion
        new_mask = new_mask + (dilated_mask - binary_mask) * weight
        new_mask = np.clip(new_mask, 0, 1)
        
        return new_mask
    
    def _shrink_region(self):
        """
        Shrink the current segmentation region
        
        Returns:
            numpy.ndarray: Updated mask
        """
        # Create a copy of the current mask
        new_mask = self.mask.copy()
        
        # Get binary mask and perform morphological erosion
        binary_mask = (new_mask > 0.5).astype(np.uint8)
        kernel_size = max(2, int(self.pointer_radius * self.shrink_factor))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)
        
        # Update the mask based on erosion, but only near the pointer
        # Create a distance-based weight centered at the pointer
        y, x = int(self.pointer_y[0]), int(self.pointer_x[0])
        y_grid, x_grid = np.ogrid[:self.height, :self.width]
        distance = np.sqrt((y_grid - y)**2 + (x_grid - x)**2)
        influence_radius = self.pointer_radius * 3
        weight = np.exp(-0.5 * (distance / influence_radius)**2)
        
        # Apply weighted shrinking
        new_mask = new_mask - (binary_mask - eroded_mask) * weight
        new_mask = np.clip(new_mask, 0, 1)
        
        return new_mask
    
    def _calculate_dice(self):
        """
        Calculate Dice coefficient between current mask and ground truth
        
        Returns:
            float: Dice coefficient
        """
        return dice_coefficient(self.mask, self.ground_truth)
    
    def _calculate_boundary_overlap(self):
        """
        Calculate boundary overlap between current mask and ground truth
        
        Returns:
            float: Boundary overlap score
        """
        # Extract boundaries from mask and ground truth
        mask_boundary = cv2.Canny((self.mask > 0.5).astype(np.uint8) * 255, 100, 200)
        gt_boundary = cv2.Canny((self.ground_truth > 0.5).astype(np.uint8) * 255, 100, 200)
        
        # Calculate overlap
        intersection = np.logical_and(mask_boundary, gt_boundary).sum()
        union = np.logical_or(mask_boundary, gt_boundary).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_exploration_reward(self):
        """
        Calculate exploration reward based on visited areas
        
        Returns:
            float: Exploration reward
        """
        # Calculate total visited area relative to the mask area
        mask_area = max(1, np.sum(self.mask > 0.5))
        visited_area = np.sum(self.visited_map > 0)
        
        exploration_ratio = min(1.0, visited_area / (3 * mask_area))
        return exploration_ratio
    
    def _calculate_reward(self, prev_dice, new_dice, action, valid_action):
        """
        Calculate reward based on improvement in Dice coefficient and other factors
        
        Args:
            prev_dice (float): Previous Dice coefficient
            new_dice (float): New Dice coefficient
            action (int): Action taken
            valid_action (bool): Whether the action was valid
            
        Returns:
            float: Reward
        """
        # Initialize reward components
        dice_reward = 0.0
        boundary_reward = 0.0
        step_penalty = self.reward_weights['step_penalty']
        invalid_action_penalty = 0.0
        completion_bonus = 0.0
        exploration_reward = 0.0
        
        # Calculate dice improvement
        dice_improvement = new_dice - prev_dice
        
        # Dense rewards based on improvement in Dice coefficient
        if self.dense_rewards:
            dice_reward = dice_improvement * 10.0  # Scale up small improvements
        else:
            # Sparse rewards only for significant improvements
            if dice_improvement > 0.01:
                dice_reward = dice_improvement * 5.0
        
        # Boundary overlap reward
        boundary_overlap = self._calculate_boundary_overlap()
        boundary_reward = boundary_overlap * self.reward_weights['boundary']
        
        # Invalid action penalty
        if not valid_action:
            invalid_action_penalty = self.reward_weights['invalid_penalty']
        
        # Completion bonus for the confirm action
        if action == 6:  # Confirm action
            completion_bonus = self.reward_weights['completion_bonus'] * new_dice
        
        # Exploration reward
        if self.reward_weights['exploration'] > 0:
            exploration_reward = self._calculate_exploration_reward() * self.reward_weights['exploration']
        
        # Calculate total reward
        reward = (
            dice_reward * self.reward_weights['dice'] +
            boundary_reward +
            step_penalty +
            invalid_action_penalty +
            completion_bonus +
            exploration_reward
        )
        
        return reward
    
    def render(self, mode='rgb_array', show=False):
        """
        Render the environment
        
        Args:
            mode (str): Rendering mode
            show (bool): Whether to show the rendered image
            
        Returns:
            numpy.ndarray: Rendered image
        """
        # Create a copy of the image
        img = self.image.copy()
        
        # Create visualization mask with transparency
        mask_viz = np.zeros_like(img, dtype=np.uint8)
        mask_viz[self.mask > 0.5] = [0, 255, 0]  # Green for current mask
        
        # Overlay ground truth for comparison
        gt_viz = np.zeros_like(img, dtype=np.uint8)
        gt_viz[self.ground_truth > 0.5] = [255, 0, 0]  # Red for ground truth
        
        # Create a combined visualization
        alpha = 0.4
        combined = cv2.addWeighted(img, 1.0, mask_viz, alpha, 0)
        combined = cv2.addWeighted(combined, 1.0, gt_viz, alpha * 0.5, 0)
        
        # Draw pointer
        pointer_x, pointer_y = int(self.pointer_x[0]), int(self.pointer_y[0])
        cv2.circle(combined, (pointer_x, pointer_y), self.pointer_radius, (0, 0, 255), 2)
        
        # Add text with metrics
        dice = self._calculate_dice()
        iou = iou_score(self.mask, self.ground_truth)
        
        text_info = f"Steps: {self.step_count}/{self.max_steps}, Dice: {dice:.4f}, IoU: {iou:.4f}"
        cv2.putText(combined, text_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if show:
            plt.figure(figsize=(10, 8))
            plt.imshow(combined)
            
            # Add a visualization of action history
            action_names = ["Up", "Down", "Left", "Right", "Expand", "Shrink", "Confirm"]
            action_history_str = ", ".join([action_names[a] for a in self.action_history[-10:]])
            plt.title(f"Last actions: {action_history_str}")
            
            plt.axis('off')
            plt.show()
        
        return combined
    
    def close(self):
        """
        Close the environment
        """
        pass


def create_env_factory(images, ground_truths, **kwargs):
    """
    Create a factory function for generating environments
    
    Args:
        images (list): List of images
        ground_truths (list): List of ground truth masks
        **kwargs: Additional arguments for environment initialization
        
    Returns:
        function: Environment factory function
    """
    def env_factory(index=None):
        if index is None:
            # Randomly select an image and ground truth
            index = np.random.randint(0, len(images))
        
        # Create environment for the selected image and ground truth
        image = images[index]
        ground_truth = ground_truths[index]
        
        return EnhancedPolypSegmentationEnv(image, ground_truth, **kwargs)
    
    return env_factory 