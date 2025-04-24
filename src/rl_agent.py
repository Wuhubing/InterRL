import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """
    Policy network for PPO agent
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the policy network
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            output_dim (int): Output dimension (number of actions)
        """
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Action probabilities
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return F.softmax(x, dim=-1)

class ValueNetwork(nn.Module):
    """
    Value network for PPO agent
    """
    
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize the value network
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
        """
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: State value
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class PPOAgent:
    """
    PPO agent for interactive polyp segmentation
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, feature_extractor, device='cpu', lr=3e-4, gamma=0.99, clip_ratio=0.2, entropy_coef=0.01):
        """
        Initialize the PPO agent
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            output_dim (int): Output dimension (number of actions)
            feature_extractor (PolypFeatureExtractor): Feature extractor
            device (str): Device to use for computations
            lr (float): Learning rate
            gamma (float): Discount factor
            clip_ratio (float): PPO clip ratio
            entropy_coef (float): Entropy coefficient
        """
        self.device = device
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.feature_extractor = feature_extractor
        
        # Initialize networks
        self.policy_network = PolicyNetwork(input_dim, hidden_dim, output_dim).to(device)
        self.value_network = ValueNetwork(input_dim, hidden_dim).to(device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)
        
        # Initialize memory
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def get_action(self, state, eval_mode=False):
        """
        Get action from policy network
        
        Args:
            state (dict): Environment state
            eval_mode (bool): Evaluation mode
            
        Returns:
            tuple: Action, log probability, value
        """
        # Extract features from state
        features = self.feature_extractor.extract_features(state)
        
        # Get action probabilities
        action_probs = self.policy_network(features)
        
        # Get state value
        value = self.value_network(features)
        
        # Sample action from distribution
        if eval_mode:
            action = torch.argmax(action_probs, dim=1).item()
            log_prob = torch.log(action_probs[0, action])
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action = action.item()
        
        return action, log_prob, value.item()
    
    def remember(self, state, action, log_prob, reward, value, done):
        """
        Store experience in memory
        
        Args:
            state (dict): Environment state
            action (int): Action taken
            log_prob (float): Log probability of action
            reward (float): Reward received
            value (float): State value
            done (bool): Done flag
        """
        features = self.feature_extractor.extract_features(state)
        
        self.states.append(features)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns(self, next_value, normalize=True):
        """
        Compute returns and advantages
        
        Args:
            next_value (float): Next state value
            normalize (bool): Whether to normalize advantages
            
        Returns:
            tuple: Returns, advantages
        """
        returns = []
        advantages = []
        
        gae = 0
        
        # Convert rewards and values to tensors
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)
        values = torch.tensor(self.values + [next_value], dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(self.device)
        
        # Compute Generalized Advantage Estimation (GAE)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        # Convert advantages to tensor
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # Compute returns
        returns = advantages + torch.tensor(self.values, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self, next_value, epochs=4, batch_size=32):
        """
        Update policy and value networks
        
        Args:
            next_value (float): Next state value
            epochs (int): Number of epochs
            batch_size (int): Batch size
            
        Returns:
            tuple: Policy loss, value loss, entropy
        """
        # Compute returns and advantages
        returns, advantages = self.compute_returns(next_value)
        
        # Convert actions and log_probs to tensors
        actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(self.device)
        
        # Convert states to tensor
        states = torch.cat(self.states, dim=0)
        
        # Get number of samples
        n_samples = len(self.states)
        
        # Initialize losses
        policy_loss_epoch = 0
        value_loss_epoch = 0
        entropy_epoch = 0
        
        # Update networks for multiple epochs
        for _ in range(epochs):
            # Generate random indices
            indices = np.random.permutation(n_samples)
            
            # Split indices into batches
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Get action probabilities and values
                batch_action_probs = self.policy_network(batch_states)
                batch_values = self.value_network(batch_states).squeeze()
                
                # Calculate log probabilities, entropy
                dist = Categorical(batch_action_probs)
                batch_new_log_probs = dist.log_prob(batch_actions)
                batch_entropy = dist.entropy().mean()
                
                # Compute ratio (policy / old policy)
                ratio = torch.exp(batch_new_log_probs - batch_old_log_probs)
                
                # Compute PPO loss
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(batch_values, batch_returns)
                
                # Compute total loss
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * batch_entropy
                
                # Update policy and value networks
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                # Update losses
                policy_loss_epoch += policy_loss.item()
                value_loss_epoch += value_loss.item()
                entropy_epoch += batch_entropy.item()
        
        # Compute average losses
        n_batches = (n_samples + batch_size - 1) // batch_size
        policy_loss_epoch /= (epochs * n_batches)
        value_loss_epoch /= (epochs * n_batches)
        entropy_epoch /= (epochs * n_batches)
        
        # Clear memory
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        return policy_loss_epoch, value_loss_epoch, entropy_epoch
    
    def save(self, path):
        """
        Save model
        
        Args:
            path (str): Path to save model
        """
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """
        Load model
        
        Args:
            path (str): Path to load model
        """
        checkpoint = torch.load(path)
        
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict']) 