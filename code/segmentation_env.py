import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import random
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage import graph
from skimage.color import rgb2gray
import gym
from gym import spaces

class SegmentationEnv(gym.Env):
    """
    Interactive Segmentation Environment for reinforcement learning
    
    This environment simulates an interactive segmentation process where an
    agent selects regions to add or remove from a segmentation mask.
    """
    def __init__(
        self,
        image=None,
        gt_mask=None,
        max_steps=20,
        n_segments=100,
        compactness=10,
        sigma=1,
        reward_type='dice',
        device=torch.device('cpu')
    ):
        """
        Initialize segmentation environment
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            gt_mask: Ground truth mask as numpy array (H, W)
            max_steps: Maximum number of steps per episode
            n_segments: Number of superpixels to generate
            compactness: Compactness parameter for SLIC
            sigma: Sigma parameter for SLIC
            reward_type: Type of reward ('dice' or 'iou')
            device: Device to use for computations
        """
        super(SegmentationEnv, self).__init__()
        
        self.device = device
        self.max_steps = max_steps
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.reward_type = reward_type
        
        # Initialize image and ground truth
        self.image = None
        self.gt_mask = None
        self.superpixels = None
        self.superpixel_properties = None
        self.current_mask = None
        self.adjacency_matrix = None
        self.region_adj_graph = None
        
        # Initialize episode variables
        self.step_count = 0
        self.previous_score = 0
        
        # Load image and ground truth if provided
        if image is not None and gt_mask is not None:
            self.load_image_and_gt(image, gt_mask)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(n_segments + 1)  # n_segments actions + 1 for "done"
        
        # Observation space: [image, current mask, gt regions (for cheating version), previous action]
        # Image shape: (H, W, 3), mask shape: (H, W, 1), gt regions: (n_segments, 1), previous action: (1, )
        # Total channels: 3 + 1 + 1 + 1 = 6
        # For the initial implementation, we'll use (H, W, 6) as observation space
        if image is not None:
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(image.shape[0], image.shape[1], 6),
                dtype=np.float32
            )
        else:
            # Default observation space if no image is provided
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(288, 384, 6),
                dtype=np.float32
            )
    
    def load_image_and_gt(self, image, gt_mask):
        """
        Load image and ground truth mask
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            gt_mask: Ground truth mask as numpy array (H, W)
        """
        # Convert to numpy arrays if necessary
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.squeeze().cpu().numpy()
        
        # Store image and gt_mask
        self.image = image
        self.gt_mask = gt_mask > 0.5  # Convert to boolean mask
        
        # Generate superpixels
        self._generate_superpixels()
        
        # Initialize current mask
        self.current_mask = np.zeros_like(self.gt_mask, dtype=bool)
        
        # Update observation space
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.image.shape[0], self.image.shape[1], 6),
            dtype=np.float32
        )
    
    def _generate_superpixels(self):
        """Generate superpixels and compute their properties"""
        # Generate superpixels using SLIC
        self.superpixels = slic(
            self.image,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=self.sigma,
            start_label=0
        )
        
        # Create region adjacency graph
        self.region_adj_graph = graph.RAG(self.superpixels)
        
        # Compute adjacency matrix
        n_superpixels = self.superpixels.max() + 1
        self.adjacency_matrix = np.zeros((n_superpixels, n_superpixels), dtype=bool)
        
        for edge in self.region_adj_graph.edges():
            self.adjacency_matrix[edge[0], edge[1]] = True
            self.adjacency_matrix[edge[1], edge[0]] = True
        
        # Compute superpixel properties
        regions = regionprops(self.superpixels + 1)  # Add 1 to avoid region with label 0
        
        self.superpixel_properties = []
        for i in range(len(regions)):
            region = regions[i]
            
            # Calculate mean color
            mask = self.superpixels == i
            mean_color = np.mean(self.image[mask], axis=0)
            
            # Calculate overlap with ground truth
            gt_overlap = np.sum(self.gt_mask & mask) / np.sum(mask)
            
            self.superpixel_properties.append({
                'label': i,
                'centroid': region.centroid,
                'area': region.area,
                'mean_color': mean_color,
                'gt_overlap': gt_overlap,
                'mask': mask
            })
    
    def reset(self):
        """Reset environment and return initial observation
        
        Returns:
            Observation: Initial state observation
            Info: Additional information
        """
        # Reset episode variables
        self.step_count = 0
        self.current_mask = np.zeros_like(self.gt_mask, dtype=bool)
        self.previous_score = 0
        self.previous_action = -1  # No previous action
        
        # Calculate initial metrics
        info = self._calculate_metrics()
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation, info
    
    def step(self, action):
        """Take a step in the environment
        
        Args:
            action: Action to take (superpixel index or "done" action)
            
        Returns:
            observation: Next state observation
            reward: Reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        # Increment step count
        self.step_count += 1
        
        # Check if episode is done
        if action == self.n_segments or self.step_count >= self.max_steps:
            # Agent chose to end episode or max steps reached
            done = True
        else:
            # Toggle the selected superpixel in the mask
            mask = self.superpixel_properties[action]['mask']
            self.current_mask[mask] = ~self.current_mask[mask]
            done = False
        
        # Calculate metrics and reward
        info = self._calculate_metrics()
        reward = self._calculate_reward(info)
        
        # Update previous action
        self.previous_action = action
        
        # Update previous score
        self.previous_score = info['dice'] if self.reward_type == 'dice' else info['iou']
        
        # Get next observation
        observation = self._get_observation()
        
        return observation, reward, done, info
    
    def _get_observation(self):
        """Get current observation
        
        Returns:
            observation: Current state observation
        """
        # Initialize observation channels
        observation = np.zeros((self.image.shape[0], self.image.shape[1], 6), dtype=np.float32)
        
        # Image channels (0-2)
        observation[:, :, :3] = self.image
        
        # Current mask channel (3)
        observation[:, :, 3] = self.current_mask
        
        # Ground truth mask channel (4) - only for "cheating" agent
        observation[:, :, 4] = self.gt_mask
        
        # Previous action channel (5)
        if self.previous_action >= 0 and self.previous_action < self.n_segments:
            observation[:, :, 5] = self.superpixel_properties[self.previous_action]['mask']
        
        return observation
    
    def _calculate_metrics(self):
        """Calculate segmentation metrics
        
        Returns:
            info: Dictionary of metrics
        """
        # Calculate intersection and union
        intersection = np.sum(self.current_mask & self.gt_mask)
        union = np.sum(self.current_mask | self.gt_mask)
        
        # Calculate Dice coefficient
        dice = (2 * intersection) / (np.sum(self.current_mask) + np.sum(self.gt_mask) + 1e-8)
        
        # Calculate IoU
        iou = intersection / (union + 1e-8)
        
        return {
            'dice': dice,
            'iou': iou,
            'intersection': intersection,
            'union': union,
            'n_pixels_mask': np.sum(self.current_mask),
            'n_pixels_gt': np.sum(self.gt_mask),
            'step': self.step_count
        }
    
    def _calculate_reward(self, info):
        """Calculate reward for current state
        
        Args:
            info: Dictionary of metrics
            
        Returns:
            reward: Calculated reward
        """
        # Calculate current score
        current_score = info['dice'] if self.reward_type == 'dice' else info['iou']
        
        # Calculate reward as improvement in score
        reward = current_score - self.previous_score
        
        # Add bonus for finishing with good segmentation
        if self.step_count >= self.max_steps or self.previous_action == self.n_segments:
            if current_score > 0.85:  # Bonus for excellent segmentation
                reward += 1.0
            elif current_score > 0.7:  # Bonus for good segmentation
                reward += 0.5
        
        return reward
    
    def render(self, mode='rgb_array', filename=None):
        """Render the environment
        
        Args:
            mode: Rendering mode
            filename: Filename to save the rendering
            
        Returns:
            Image or None depending on mode
        """
        # Create figure and subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        axs[0].imshow(self.image)
        axs[0].set_title('Original Image')
        axs[0].axis('off')
        
        # Plot current mask
        axs[1].imshow(self.current_mask, cmap='gray')
        axs[1].set_title(f'Current Mask (Dice: {self._calculate_metrics()["dice"]:.3f})')
        axs[1].axis('off')
        
        # Plot ground truth mask
        axs[2].imshow(self.gt_mask, cmap='gray')
        axs[2].set_title('Ground Truth')
        axs[2].axis('off')
        
        plt.tight_layout()
        
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
            return None
        elif mode == 'rgb_array':
            # Convert figure to numpy array
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close()
            return img
        elif mode == 'human':
            plt.show()
            return None
    
    def render_interaction_sequence(self, actions, filename=None):
        """Render sequence of interactions
        
        Args:
            actions: List of actions
            filename: Filename to save the rendering
            
        Returns:
            Image or None depending on mode
        """
        # Reset environment
        self.reset()
        
        # Create figure and subplots
        n_steps = min(len(actions), 10)  # Show at most 10 steps
        fig, axs = plt.subplots(3, n_steps + 1, figsize=(3 * (n_steps + 1), 9))
        
        # Plot initial state
        axs[0, 0].imshow(self.image)
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')
        
        axs[1, 0].imshow(self.current_mask, cmap='gray')
        axs[1, 0].set_title('Initial Mask')
        axs[1, 0].axis('off')
        
        axs[2, 0].imshow(self.gt_mask, cmap='gray')
        axs[2, 0].set_title('Ground Truth')
        axs[2, 0].axis('off')
        
        # Execute actions and plot results
        for i, action in enumerate(actions[:n_steps]):
            # Take step
            observation, reward, done, info = self.step(action)
            
            # Plot results
            axs[0, i + 1].imshow(self.image)
            if action < self.n_segments:
                action_mask = self.superpixel_properties[action]['mask']
                highlighted_image = self.image.copy()
                highlighted_image[action_mask] = [1, 0, 0]  # Red highlight
                axs[0, i + 1].imshow(highlighted_image)
                axs[0, i + 1].set_title(f'Step {i+1}: Region {action}')
            else:
                axs[0, i + 1].set_title(f'Step {i+1}: Done')
            axs[0, i + 1].axis('off')
            
            axs[1, i + 1].imshow(self.current_mask, cmap='gray')
            axs[1, i + 1].set_title(f'Dice: {info["dice"]:.3f}')
            axs[1, i + 1].axis('off')
            
            axs[2, i + 1].imshow(np.abs(self.current_mask.astype(int) - self.gt_mask.astype(int)), cmap='gray')
            axs[2, i + 1].set_title(f'Error')
            axs[2, i + 1].axis('off')
            
            if done:
                break
        
        plt.tight_layout()
        
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
            return None
        else:
            plt.show()
            return None
    
    def close(self):
        """Close environment"""
        plt.close()

class SegmentationEnvDataset:
    """Dataset of segmentation environments
    
    This class creates a dataset of segmentation environments from
    a directory of images and a directory of ground truth masks.
    """
    def __init__(
        self,
        image_dir,
        mask_dir,
        n_segments=100,
        compactness=10,
        sigma=1,
        reward_type='dice',
        max_steps=20,
        device=torch.device('cpu')
    ):
        """
        Initialize segmentation environment dataset
        
        Args:
            image_dir: Directory of images
            mask_dir: Directory of ground truth masks
            n_segments: Number of superpixels to generate
            compactness: Compactness parameter for SLIC
            sigma: Sigma parameter for SLIC
            reward_type: Type of reward ('dice' or 'iou')
            max_steps: Maximum number of steps per episode
            device: Device to use for computations
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.reward_type = reward_type
        self.max_steps = max_steps
        self.device = device
        
        # Get list of image and mask files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.jpg') or f.endswith('.png')])
        
        # Check that there are matching files
        if len(self.image_files) != len(self.mask_files):
            print(f"Warning: Number of image files ({len(self.image_files)}) does not match number of mask files ({len(self.mask_files)})")
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
    
    def __len__(self):
        """Get length of dataset"""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get segmentation environment for specified index
        
        Args:
            idx: Index of environment to get
            
        Returns:
            env: Segmentation environment
        """
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        image = np.array(Image.open(image_path)) / 255.0
        mask = np.array(Image.open(mask_path).convert('L')) / 255.0
        
        # Create environment
        env = SegmentationEnv(
            image=image,
            gt_mask=mask,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=self.sigma,
            reward_type=self.reward_type,
            max_steps=self.max_steps,
            device=self.device
        )
        
        return env 