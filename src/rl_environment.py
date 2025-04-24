import numpy as np
import torch
import gym
from gym import spaces
import cv2
from skimage import draw
import matplotlib.pyplot as plt

class PolypSegmentationEnv(gym.Env):
    """
    RL Environment for interactive polyp segmentation
    
    State: Image and current segmentation mask
    Action: Move pointer, expand region, shrink region, confirm segmentation
    Reward: Improvement in Dice coefficient
    """
    
    def __init__(self, image, gt_mask, max_steps=100, step_size=10, device='cpu'):
        """
        Initialize the environment
        
        Args:
            image (numpy.ndarray or torch.Tensor): Input image (H, W, C) or (C, H, W)
            gt_mask (numpy.ndarray or torch.Tensor): Ground truth mask (H, W) or (1, H, W)
            max_steps (int): Maximum number of steps
            step_size (int): Step size for pointer movement
            device (str): Device to use for computations
        """
        super(PolypSegmentationEnv, self).__init__()
        
        # Convert PyTorch tensors to NumPy arrays if necessary
        if isinstance(image, torch.Tensor):
            # If tensor in (C, H, W) format, convert to (H, W, C)
            if image.dim() == 3 and image.shape[0] <= 3:
                image = image.permute(1, 2, 0).cpu().numpy()
            else:
                image = image.cpu().numpy()

        if isinstance(gt_mask, torch.Tensor):
            # If tensor in (1, H, W) format, convert to (H, W)
            if gt_mask.dim() == 3 and gt_mask.shape[0] == 1:
                gt_mask = gt_mask.squeeze(0).cpu().numpy()
            else:
                gt_mask = gt_mask.cpu().numpy()
                
        self.image = image
        self.gt_mask = gt_mask.astype(np.float32)
        self.max_steps = max_steps
        self.step_size = step_size
        self.device = device
        
        # Image dimensions
        self.height, self.width = self.image.shape[:2]
        
        # Define action space
        # 0-3: Move pointer (up, down, left, right)
        # 4: Expand region
        # 5: Shrink region
        # 6: Confirm segmentation
        self.action_space = spaces.Discrete(7)
        
        # Define observation space
        # Image + current mask + pointer location
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8),
            'mask': spaces.Box(low=0, high=1, shape=(self.height, self.width), dtype=np.float32),
            'pointer_x': spaces.Box(low=0, high=self.width-1, shape=(1,), dtype=np.int32),
            'pointer_y': spaces.Box(low=0, high=self.height-1, shape=(1,), dtype=np.int32)
        })
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """
        Reset the environment
        
        Returns:
            dict: Initial observation
        """
        # Reset mask to zeros (empty)
        self.current_mask = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Initialize pointer at the center of the image
        self.pointer_x = self.width // 2
        self.pointer_y = self.height // 2
        
        # Reset steps
        self.steps = 0
        
        # Reset previous reward
        self.prev_dice = 0.0
        
        return self._get_observation()
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action (int): Action to take
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Increment step counter
        self.steps += 1
        
        # Process action
        if action == 0:  # Move up
            self.pointer_y = max(0, self.pointer_y - self.step_size)
        elif action == 1:  # Move down
            self.pointer_y = min(self.height - 1, self.pointer_y + self.step_size)
        elif action == 2:  # Move left
            self.pointer_x = max(0, self.pointer_x - self.step_size)
        elif action == 3:  # Move right
            self.pointer_x = min(self.width - 1, self.pointer_x + self.step_size)
        elif action == 4:  # Expand region
            self._expand_region()
        elif action == 5:  # Shrink region
            self._shrink_region()
        elif action == 6:  # Confirm segmentation
            pass  # Do nothing, just evaluate
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        done = (self.steps >= self.max_steps) or (action == 6)
        
        # Get observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'dice': self._calculate_dice(),
            'iou': self._calculate_iou(),
            'steps': self.steps
        }
        
        return observation, reward, done, info
    
    def _get_observation(self):
        """
        Get the current observation
        
        Returns:
            dict: Current observation
        """
        return {
            'image': self.image,
            'mask': self.current_mask,
            'pointer_x': np.array([self.pointer_x], dtype=np.int32),
            'pointer_y': np.array([self.pointer_y], dtype=np.int32)
        }
    
    def _expand_region(self, radius=10):
        """
        Expand the region around the current pointer
        
        Args:
            radius (int): Radius of the region to expand
        """
        # Create a circular region around the pointer
        rr, cc = draw.disk((self.pointer_y, self.pointer_x), radius, shape=(self.height, self.width))
        self.current_mask[rr, cc] = 1.0
    
    def _shrink_region(self, radius=10):
        """
        Shrink the region around the current pointer
        
        Args:
            radius (int): Radius of the region to shrink
        """
        # Create a circular region around the pointer
        rr, cc = draw.disk((self.pointer_y, self.pointer_x), radius, shape=(self.height, self.width))
        self.current_mask[rr, cc] = 0.0
    
    def _calculate_dice(self):
        """
        Calculate Dice coefficient between current mask and ground truth
        
        Returns:
            float: Dice coefficient
        """
        y_pred = self.current_mask.flatten()
        y_true = self.gt_mask.flatten()
        
        intersection = np.sum(y_pred * y_true)
        dice = (2. * intersection + 1e-6) / (np.sum(y_pred) + np.sum(y_true) + 1e-6)
        
        return dice
    
    def _calculate_iou(self):
        """
        Calculate IoU between current mask and ground truth
        
        Returns:
            float: IoU score
        """
        y_pred = self.current_mask.flatten()
        y_true = self.gt_mask.flatten()
        
        intersection = np.sum(y_pred * y_true)
        union = np.sum(y_pred) + np.sum(y_true) - intersection
        
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        return iou
    
    def _calculate_reward(self):
        """
        Calculate reward based on improvement in Dice coefficient
        
        Returns:
            float: Reward
        """
        current_dice = self._calculate_dice()
        reward = current_dice - self.prev_dice
        
        # Penalize if no improvement
        if reward <= 0:
            reward = -0.01
            
        # Bonus for good matches
        if current_dice > 0.9:
            reward += 1.0
        
        # Update previous dice
        self.prev_dice = current_dice
        
        return reward
    
    def render(self, mode='human'):
        """
        Render the current state
        
        Args:
            mode (str): Rendering mode
        """
        if mode == 'human':
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(self.image)
            plt.scatter(self.pointer_x, self.pointer_y, c='red', marker='x', s=100)
            plt.title('Image with Pointer')
            plt.axis('off')
            
            # Current mask
            plt.subplot(1, 3, 2)
            plt.imshow(self.current_mask, cmap='gray')
            plt.title(f'Current Mask (Dice: {self._calculate_dice():.4f})')
            plt.axis('off')
            
            # Ground truth
            plt.subplot(1, 3, 3)
            plt.imshow(self.gt_mask, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return None

class PolypFeatureExtractor:
    """
    Extract features from the environment state for the RL agent
    """
    
    def __init__(self, cnn_model=None, device='cpu'):
        """
        Initialize the feature extractor
        
        Args:
            cnn_model (torch.nn.Module, optional): CNN model for feature extraction
            device (str): Device to use for computations
        """
        self.cnn_model = cnn_model
        self.device = device
        
    def extract_features(self, observation):
        """
        Extract features from the observation
        
        Args:
            observation (dict): Environment observation
            
        Returns:
            torch.Tensor: Extracted features
        """
        # Get the image, mask, and pointer from observation
        image = observation['image']
        mask = observation['mask']
        pointer_x = observation['pointer_x'][0]
        pointer_y = observation['pointer_y'][0]
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        
        # Stack image and mask
        state = np.concatenate([image, mask[..., np.newaxis]], axis=2)
        
        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        if self.cnn_model is not None:
            # Extract features using CNN
            with torch.no_grad():
                features = self.cnn_model(state_tensor)
        else:
            # Resize to a fixed size if no CNN model is provided
            features = torch.nn.functional.adaptive_avg_pool2d(state_tensor, (16, 16))
            features = features.view(1, -1)
        
        # Append pointer location
        pointer = torch.tensor([[
            pointer_x / float(image.shape[1]),  # Normalize x coordinate
            pointer_y / float(image.shape[0])   # Normalize y coordinate
        ]], dtype=torch.float32).to(self.device)
        
        # Concatenate features and pointer
        final_features = torch.cat([features, pointer], dim=1)
        
        return final_features 