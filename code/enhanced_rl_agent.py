import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import os
import sys
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Optional, Union, Any

# Add code directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.unet_model import UNet

# Experience tuple for storing transitions
Experience = namedtuple('Experience', ['obs', 'action', 'reward', 'next_obs', 'done'])

class EnhancedFeatureExtractor(nn.Module):
    """Enhanced feature extractor that can use either UNet encoder or a CNN"""
    
    def __init__(self, unet_encoder=None, device='cuda', use_cnn=False):
        super(EnhancedFeatureExtractor, self).__init__()
        self.device = device
        self.use_cnn = use_cnn
        self.unet_encoder = unet_encoder
        
        # If UNet encoder is not provided or we explicitly want to use CNN
        if unet_encoder is None or use_cnn:
            # Use a CNN feature extractor
            self.conv1 = nn.Conv2d(7, 32, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(256)
            self.output_dim = 1024  # Output after adaptive pooling
        else:
            # Use the UNet encoder as feature extractor
            # The output dimension depends on the UNet architecture
            self.output_dim = 512  # This should match the UNet encoder's output
        
        self.to(device)
    
    def forward(self, x):
        """Extract features from input state
        
        Args:
            x: Input state containing image, mask, and pointer information
               Shape: [batch_size, channels, height, width]
        
        Returns:
            Features extracted from the state
        """
        x = x.to(self.device)
        
        if self.use_cnn or self.unet_encoder is None:
            # CNN feature extraction
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            
            # Global average pooling followed by max pooling
            avg_pool = F.adaptive_avg_pool2d(x, (2, 2))
            max_pool = F.adaptive_max_pool2d(x, (2, 2))
            
            # Combine and flatten
            x = torch.cat([avg_pool, max_pool], dim=1)
            x = x.view(x.size(0), -1)
        else:
            # Use UNet encoder for feature extraction
            with torch.no_grad():
                x1, x2, x3, x4, x5 = self.unet_encoder(x[:, :3, :, :])  # Extract RGB features
            
            # Combine features from different levels
            # Apply average pooling to reduce dimensions
            x1 = F.adaptive_avg_pool2d(x1, (1, 1)).view(x.size(0), -1)
            x2 = F.adaptive_avg_pool2d(x2, (1, 1)).view(x.size(0), -1)
            x3 = F.adaptive_avg_pool2d(x3, (1, 1)).view(x.size(0), -1)
            x4 = F.adaptive_avg_pool2d(x4, (1, 1)).view(x.size(0), -1)
            x5 = F.adaptive_avg_pool2d(x5, (1, 1)).view(x.size(0), -1)
            
            # Concatenate features
            x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        
        return x

class PolicyNetwork(nn.Module):
    """Policy network for the RL agent"""
    
    def __init__(self, input_dim, hidden_dim=128, action_dim=9):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Action dimensions:
        # 0: Up, 1: Down, 2: Left, 3: Right, 4: Up-Left, 5: Up-Right, 6: Down-Left, 7: Down-Right
        # 8: Expand, 9: Shrink
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
    
    def forward(self, x):
        """Forward pass through the policy network"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs
    
    def get_action(self, state, deterministic=False):
        """Get action based on state"""
        action_probs = self.forward(state)
        
        if deterministic:
            # Choose action with highest probability
            action = torch.argmax(action_probs, dim=-1)
        else:
            # Sample action from probability distribution
            dist = Categorical(action_probs)
            action = dist.sample()
        
        log_prob = torch.log(action_probs.squeeze(0)[action])
        return action.item(), log_prob

class ValueNetwork(nn.Module):
    """Value network for the RL agent"""
    
    def __init__(self, input_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
    
    def forward(self, x):
        """Forward pass through the value network"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class ExperienceBuffer:
    """Buffer to store experiences for PPO updates"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []
    
    def store(self, state, action, reward, done, log_prob, value):
        """Store experience in buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def compute_advantages_and_returns(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Compute advantages and returns using GAE"""
        self.advantages = []
        self.returns = []
        
        next_value = last_value
        next_advantage = 0
        
        for i in reversed(range(len(self.rewards))):
            # Calculate TD error
            delta = self.rewards[i] + gamma * next_value * (1 - self.dones[i]) - self.values[i]
            
            # Calculate advantage using GAE
            advantage = delta + gamma * gae_lambda * (1 - self.dones[i]) * next_advantage
            
            # Update for next iteration
            next_value = self.values[i]
            next_advantage = advantage
            
            # Store advantage and return
            self.advantages.insert(0, advantage)
            self.returns.insert(0, advantage + self.values[i])
    
    def get_batches(self, batch_size=64):
        """Get batches of experiences for training"""
        # Convert lists to tensors
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions)
        log_probs = torch.stack(self.log_probs)
        returns = torch.tensor(self.returns)
        advantages = torch.tensor(self.advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get indices for random batches
        indices = np.arange(len(states))
        np.random.shuffle(indices)
        
        # Create batches
        for i in range(0, len(states), batch_size):
            batch_indices = indices[i:i+batch_size]
            yield (
                states[batch_indices],
                actions[batch_indices],
                log_probs[batch_indices],
                returns[batch_indices],
                advantages[batch_indices]
            )
    
    def clear(self):
        """Clear buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []
    
    def __len__(self):
        return len(self.states)

class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for images
    
    Takes an image (or stacked images) and extracts features for the policy
    """
    def __init__(self, input_shape, feature_dim=128):
        """Initialize feature extractor
        
        Args:
            input_shape: Shape of input images (channels, height, width)
            feature_dim: Dimension of output features
        """
        super().__init__()
        
        # Input shape should be (batch_size, channels, height, width)
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        
        # CNN layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate output size of CNN
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        h = input_shape[1]
        w = input_shape[2]
        
        h = conv2d_size_out(h, 8, 4)
        w = conv2d_size_out(w, 8, 4)
        
        h = conv2d_size_out(h, 4, 2)
        w = conv2d_size_out(w, 4, 2)
        
        h = conv2d_size_out(h, 3, 1)
        w = conv2d_size_out(w, 3, 1)
        
        self.fc = nn.Linear(64 * h * w, feature_dim)
        
    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            features: Output features of shape (batch_size, feature_dim)
        """
        # Normalize input if it's not already in [-1, 1]
        if x.max() > 1.0:
            x = x / 255.0
        
        # CNN forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten and pass through FC layer
        x = x.view(x.size(0), -1)
        features = F.relu(self.fc(x))
        
        return features

class PPOActor(nn.Module):
    """Actor network for PPO
    
    Takes observations and outputs action probabilities
    """
    def __init__(self, obs_shape, feature_extractor, feature_extractor_type, n_actions, hidden_size=128):
        """Initialize actor network
        
        Args:
            obs_shape: Shape of observations
            feature_extractor: Feature extractor to use (if any)
            feature_extractor_type: Type of feature extractor ('cnn' or 'unet')
            n_actions: Number of possible actions
            hidden_size: Size of hidden layers
        """
        super().__init__()
        
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.feature_extractor_type = feature_extractor_type
        
        # Use provided feature extractor or create a new one
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
            # If using a UNet, we'll extract features from a specific point
            if feature_extractor_type == 'unet':
                # We'll use the encoded features from UNet (before decoding)
                feature_dim = 512  # Typical UNet bottleneck size
            else:
                feature_dim = feature_extractor.feature_dim
        else:
            # Create a new CNN feature extractor
            self.feature_extractor = CNNFeatureExtractor(obs_shape, hidden_size)
            feature_dim = hidden_size
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
        
    def forward(self, obs):
        """Forward pass
        
        Args:
            obs: Observation tensor of shape (batch_size, channels, height, width)
            
        Returns:
            action_probs: Action probabilities of shape (batch_size, n_actions)
        """
        # Extract features
        if self.feature_extractor_type == 'unet':
            # For UNet, extract features from the bottleneck
            # This assumes the UNet's forward method returns both the output and bottleneck features
            # You might need to modify this based on your UNet implementation
            features = self.feature_extractor.inc(obs)
            features = self.feature_extractor.down1(features)
            features = self.feature_extractor.down2(features)
            features = self.feature_extractor.down3(features)
            features = self.feature_extractor.down4(features)
            # Reshape features to (batch_size, feature_dim)
            features = features.mean(dim=(2, 3))  # Global average pooling
        else:
            # Use CNN feature extractor
            features = self.feature_extractor(obs)
        
        # Compute action logits
        action_logits = self.policy_head(features)
        
        # Convert to probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs
    
    def get_action_and_log_prob(self, obs):
        """Get action and log probability
        
        Args:
            obs: Observation tensor of shape (batch_size, channels, height, width)
            
        Returns:
            action: Selected action
            log_prob: Log probability of selected action
        """
        # Get action probabilities
        action_probs = self.forward(obs)
        
        # Sample action from distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        # Compute log probability
        log_prob = action_dist.log_prob(action)
        
        return action, log_prob
    
    def get_log_prob(self, obs, action):
        """Get log probability of action
        
        Args:
            obs: Observation tensor of shape (batch_size, channels, height, width)
            action: Action tensor of shape (batch_size,)
            
        Returns:
            log_prob: Log probability of action
        """
        # Get action probabilities
        action_probs = self.forward(obs)
        
        # Create distribution
        action_dist = torch.distributions.Categorical(action_probs)
        
        # Compute log probability
        log_prob = action_dist.log_prob(action)
        
        return log_prob

class PPOCritic(nn.Module):
    """Critic network for PPO
    
    Takes observations and outputs value estimates
    """
    def __init__(self, obs_shape, feature_extractor, feature_extractor_type, hidden_size=128):
        """Initialize critic network
        
        Args:
            obs_shape: Shape of observations
            feature_extractor: Feature extractor to use (if any)
            feature_extractor_type: Type of feature extractor ('cnn' or 'unet')
            hidden_size: Size of hidden layers
        """
        super().__init__()
        
        self.obs_shape = obs_shape
        self.feature_extractor_type = feature_extractor_type
        
        # Use provided feature extractor or create a new one
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
            # If using a UNet, we'll extract features from a specific point
            if feature_extractor_type == 'unet':
                # We'll use the encoded features from UNet (before decoding)
                feature_dim = 512  # Typical UNet bottleneck size
            else:
                feature_dim = feature_extractor.feature_dim
        else:
            # Create a new CNN feature extractor
            self.feature_extractor = CNNFeatureExtractor(obs_shape, hidden_size)
            feature_dim = hidden_size
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, obs):
        """Forward pass
        
        Args:
            obs: Observation tensor of shape (batch_size, channels, height, width)
            
        Returns:
            value: Value estimate of shape (batch_size, 1)
        """
        # Extract features
        if self.feature_extractor_type == 'unet':
            # For UNet, extract features from the bottleneck
            # This assumes the UNet's forward method returns both the output and bottleneck features
            # You might need to modify this based on your UNet implementation
            features = self.feature_extractor.inc(obs)
            features = self.feature_extractor.down1(features)
            features = self.feature_extractor.down2(features)
            features = self.feature_extractor.down3(features)
            features = self.feature_extractor.down4(features)
            # Reshape features to (batch_size, feature_dim)
            features = features.mean(dim=(2, 3))  # Global average pooling
        else:
            # Use CNN feature extractor
            features = self.feature_extractor(obs)
        
        # Compute value estimate
        value = self.value_head(features)
        
        return value

class EnhancedPPOAgent:
    """Enhanced PPO agent for interactive segmentation
    
    This agent implements the PPO algorithm with some enhancements:
    - Separate actor and critic networks
    - Optional CNN feature extractor
    - Optional pretrained UNet feature extractor
    - GAE advantage estimation
    - Value function clipping
    - Entropy regularization
    """
    def __init__(
        self,
        obs_shape,
        action_space,
        hidden_size=128,
        feature_extractor=None,
        feature_extractor_type='cnn',
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device=torch.device('cpu')
    ):
        """Initialize PPO agent
        
        Args:
            obs_shape: Shape of observations
            action_space: Number of possible actions
            hidden_size: Size of hidden layers
            feature_extractor: Feature extractor to use (if any)
            feature_extractor_type: Type of feature extractor ('cnn' or 'unet')
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clip ratio
            value_coef: Value loss coefficient
            entropy_coef: Entropy loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use for computation
        """
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.device = device
        
        # PPO hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Create actor and critic networks
        self.actor = PPOActor(
            obs_shape=obs_shape,
            feature_extractor=feature_extractor,
            feature_extractor_type=feature_extractor_type,
            n_actions=action_space,
            hidden_size=hidden_size
        ).to(device)
        
        self.critic = PPOCritic(
            obs_shape=obs_shape,
            feature_extractor=feature_extractor,
            feature_extractor_type=feature_extractor_type,
            hidden_size=hidden_size
        ).to(device)
        
        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Experience buffer
        self.experience_buffer = []
        
        # Training info
        self.training_info = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': []
        }
    
    def select_action(self, obs):
        """Select action based on observation
        
        Args:
            obs: Observation tensor of shape (channels, height, width)
            
        Returns:
            action: Selected action
        """
        # Add batch dimension if necessary
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        
        # Get action probabilities
        with torch.no_grad():
            action_probs = self.actor(obs)
        
        # Sample action from distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action
    
    def store_experience(self, obs, action, reward, next_obs, done):
        """Store experience in buffer
        
        Args:
            obs: Observation tensor
            action: Action tensor
            reward: Reward tensor
            next_obs: Next observation tensor
            done: Done tensor
        """
        # Ensure tensors have proper shape for concatenation
        if isinstance(reward, torch.Tensor) and reward.dim() == 0:
            reward = reward.unsqueeze(0)  # Convert scalar to 1D tensor
            
        if isinstance(done, torch.Tensor) and done.dim() == 0:
            done = done.unsqueeze(0)  # Convert scalar to 1D tensor
            
        if isinstance(action, torch.Tensor) and action.dim() == 0:
            action = action.unsqueeze(0)  # Convert scalar to 1D tensor
            
        self.experience_buffer.append(Experience(obs, action, reward, next_obs, done))
    
    def compute_advantages(self, rewards, values, dones):
        """Compute advantages using Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Tensor of shape (batch_size,) containing rewards
            values: Tensor of shape (batch_size,) containing value estimates
            dones: Tensor of shape (batch_size,) containing done flags
            
        Returns:
            advantages: Tensor of shape (batch_size,) containing advantages
            returns: Tensor of shape (batch_size,) containing returns
        """
        # Initialize advantages and returns
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute GAE
        last_gae = 0
        for t in reversed(range(len(rewards))):
            # Compute TD error
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # Compute delta (TD error)
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # Compute advantage
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            
            # Compute return
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(self, batch_size=64, epochs=10):
        """Update agent using PPO
        
        Args:
            batch_size: Batch size for updates
            epochs: Number of epochs to train for
            
        Returns:
            info: Dictionary of training information
        """
        # Convert experiences to tensors
        obss = torch.cat([exp.obs for exp in self.experience_buffer])
        actions = torch.cat([exp.action for exp in self.experience_buffer])
        rewards = torch.cat([exp.reward for exp in self.experience_buffer])
        next_obss = torch.cat([exp.next_obs for exp in self.experience_buffer])
        dones = torch.cat([exp.done for exp in self.experience_buffer])
        
        # Compute values and advantages
        with torch.no_grad():
            values = self.critic(obss).squeeze()
            
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get old action log probabilities
        with torch.no_grad():
            old_log_probs = self.actor.get_log_prob(obss, actions)
        
        # Training loop
        for epoch in range(epochs):
            # Generate random indices for mini-batches
            indices = torch.randperm(len(obss))
            
            # Train on mini-batches
            for start_idx in range(0, len(indices), batch_size):
                # Get mini-batch indices
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Get mini-batch data
                batch_obss = obss[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Compute policy loss
                new_log_probs = self.actor.get_log_prob(batch_obss, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                values = self.critic(batch_obss).squeeze()
                value_loss = F.mse_loss(values, batch_returns)
                
                # Compute entropy loss for exploration
                action_probs = self.actor(batch_obss)
                entropy = torch.distributions.Categorical(action_probs).entropy().mean()
                entropy_loss = -entropy
                
                # Compute total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update actor
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # Store training info
                self.training_info['policy_loss'].append(policy_loss.item())
                self.training_info['value_loss'].append(value_loss.item())
                self.training_info['entropy_loss'].append(entropy_loss.item())
                self.training_info['total_loss'].append(total_loss.item())
        
        # Return mean training info
        return {
            'policy_loss': np.mean(self.training_info['policy_loss'][-10:]),
            'value_loss': np.mean(self.training_info['value_loss'][-10:]),
            'entropy_loss': np.mean(self.training_info['entropy_loss'][-10:]),
            'total_loss': np.mean(self.training_info['total_loss'][-10:])
        }
    
    def train(self):
        """Set agent to training mode"""
        self.actor.train()
        self.critic.train()
    
    def eval(self):
        """Set agent to evaluation mode"""
        self.actor.eval()
        self.critic.eval()
    
    def save(self, path):
        """Save agent to file
        
        Args:
            path: Path to save agent to
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_info': self.training_info
        }, path)
    
    def load(self, path, device=None):
        """Load agent from file
        
        Args:
            path: Path to load agent from
            device: Device to load agent to (if different from current)
        """
        if device is not None:
            self.device = device
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_info = checkpoint['training_info']
        
        # Move models to device
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

def evaluate_agent(env, agent, num_episodes=10):
    """Evaluate agent on environment
    
    Args:
        env: Environment to evaluate on
        agent: Agent to evaluate
        num_episodes: Number of episodes to evaluate for
    
    Returns:
        Tuple of (average dice score, average IoU)
    """
    dice_scores = []
    iou_scores = []
    
    for _ in range(num_episodes):
        state, info = env.reset()
        done = False
        
        while not done:
            action = agent.select_action(state, deterministic=True)
            state, _, done, info = env.step(action)
        
        dice_scores.append(info['dice'])
        iou_scores.append(info['iou'])
    
    return np.mean(dice_scores), np.mean(iou_scores) 