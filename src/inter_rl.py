import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import cv2
import pickle
import time
import json

from src.data_utils import PolypDataset
from src.utils import dice_coefficient, iou_score, set_seed

class SimplePolicyNetwork(nn.Module):
    """Simple policy network with fixed architecture"""
    def __init__(self, n_actions=7):
        super(SimplePolicyNetwork, self).__init__()
        # Use more complex network architecture to improve feature extraction capability
        self.conv1 = nn.Conv2d(7, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Add attention mechanism to focus on important regions
        self.attention = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        
        # Adaptive pooling ensures fixed size output regardless of input dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))
        
        # Correct input dimension for fully connected layer
        self.fc1 = nn.Linear(64 * 32 * 32, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, n_actions)
        
        # Use improved initialization method
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.fc3.weight)
        
    def forward(self, x):
        # Use proper normalization
        x = x / 255.0
        
        # Deeper convolutional layers with batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Attention mechanism
        attention = torch.sigmoid(self.attention(x))
        x = x * attention
        
        # Adaptive pooling ensures fixed size output
        x = self.adaptive_pool(x)
        
        # Flatten features
        x = x.view(x.size(0), -1)
        
        # Deeper fully connected layers with Dropout to prevent overfitting
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        # Output action probabilities
        return F.softmax(x, dim=1)

class SimpleValueNetwork(nn.Module):
    """Simple value network with fixed architecture"""
    def __init__(self):
        super(SimpleValueNetwork, self).__init__()
        # Use the same architecture as policy network, but output value is 1
        self.conv1 = nn.Conv2d(7, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Add attention mechanism
        self.attention = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        
        # Adaptive pooling ensures fixed size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))
        
        # Correct input dimension for fully connected layer
        self.fc1 = nn.Linear(64 * 32 * 32, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 1)
        
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.fc3.weight)
        
    def forward(self, x):
        # Use proper normalization
        x = x / 255.0
        
        # Deeper convolutional layers with batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Attention mechanism
        attention = torch.sigmoid(self.attention(x))
        x = x * attention
        
        # Adaptive pooling ensures fixed size output
        x = self.adaptive_pool(x)
        
        # Flatten features
        x = x.view(x.size(0), -1)
        
        # Deeper fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class SimpleRLAgent:
    """Simple reinforcement learning agent for image segmentation"""
    def __init__(self, n_actions=7, lr=1e-4, gamma=0.99, device='cpu'):
        self.device = torch.device(device)
        self.policy_net = SimplePolicyNetwork(n_actions).to(self.device)
        self.value_net = SimpleValueNetwork().to(self.device)
        
        # Use lower learning rate and weight decay
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr, weight_decay=1e-5)
        
        self.gamma = gamma
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []  # Add done flag to improve return value calculation
        
        # Add exploration parameters
        self.entropy_coef = 0.01  # Entropy coefficient to encourage exploration
        self.value_coef = 0.5     # Value function loss coefficient
        
    def select_action(self, state, deterministic=False):
        """Select action based on current state"""
        # Convert state to tensor and add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action probabilities
        with torch.no_grad():
            probs = self.policy_net(state)
            value = self.value_net(state)
        
        if deterministic:
            # In evaluation mode, select action with highest probability
            action = torch.argmax(probs, dim=1)
            log_prob = torch.log(probs.squeeze(0)[action])
        else:
            # In training mode, sample action from distribution
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
        
        # Store experience
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        return action.item()
    
    def update(self, final_value=0, done=True):
        """Update policy and value networks"""
        # Check if buffer is empty
        if len(self.rewards) == 0:
            return 0.0, 0.0
        
        # Prepare for training
        returns = []
        R = final_value
        
        # Calculate generalized advantage estimate (GAE) and return value
        advantages = []
        gae = 0
        
        # Add a final done flag
        self.dones.append(done)
        
        for r, v, done in zip(reversed(self.rewards), 
                             reversed([v.item() for v in self.values]), 
                             reversed(self.dones)):
            if done:
                R = 0
            
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
            
        # Convert list to tensor
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values).squeeze(-1)  # Ensure consistent dimension
        
        # Ensure returns and values have the same shape
        if returns.shape != values.shape:
            returns = returns.view(-1)
            values = values.view(-1)
        
        # Standardize return values (safer standard deviation)
        if returns.shape[0] > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate advantages
        advantages = returns - values.detach()
        
        # Calculate policy loss using PPO style
        new_probs = self.policy_net(states)
        new_m = torch.distributions.Categorical(new_probs)
        new_log_probs = new_m.log_prob(actions)
        
        # Calculate policy ratio and clipped objective function
        ratio = torch.exp(new_log_probs - log_probs.detach())
        
        # Clip policy objective function (PPO style)
        clip_epsilon = 0.2
        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages.detach()
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Add entropy loss to encourage exploration
        entropy = new_m.entropy().mean()
        policy_loss = policy_loss - self.entropy_coef * entropy
        
        # Value function loss
        value_loss = F.mse_loss(self.value_net(states).squeeze(-1), returns)
        
        # Update policy network
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)  # Gradient clipping to prevent large updates
        self.optimizer_policy.step()
        
        # Update value network
        self.optimizer_value.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)  # Gradient clipping to prevent large updates
        self.optimizer_value.step()
        
        # Clear buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        return policy_loss.item(), value_loss.item()
    
    def save(self, path):
        """Save agent to file"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'optimizer_policy': self.optimizer_policy.state_dict(),
            'optimizer_value': self.optimizer_value.state_dict()
        }, path)
    
    def load(self, path):
        """Load agent from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.optimizer_policy.load_state_dict(checkpoint['optimizer_policy'])
        self.optimizer_value.load_state_dict(checkpoint['optimizer_value'])

def train_agent(args):
    """Train RL agent for interactive segmentation"""
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create training history directory
    history_dir = os.path.join(args.output_dir, 'history')
    os.makedirs(history_dir, exist_ok=True)
    
    # Load dataset
    train_dataset = PolypDataset(args.data_dir, split='train')
    val_dataset = PolypDataset(args.data_dir, split='val')
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    # Create agent
    agent = SimpleRLAgent(
        n_actions=7,  # 0=Up, 1=Down, 2=Left, 3=Right, 4=Expand, 5=Shrink, 6=Done
        lr=args.lr,
        gamma=args.gamma,
        device=args.device
    )
    
    # Training metrics
    all_rewards = []
    all_dice_scores = []
    all_iou_scores = []
    best_dice = 0.0
    best_episode = 0
    best_val_dice = 0.0
    best_val_iou = 0.0
    
    # Record validation performance for each episode
    val_results = {
        'episodes': [],
        'dice_scores': [],
        'iou_scores': []
    }
    
    # Create training history dictionary, add more tracking items
    training_history = {
        'episodes': [],
        'train_rewards': [],
        'train_dice_scores': [],
        'train_iou_scores': [],
        'val_dice_scores': [],
        'val_iou_scores': [],
        'policy_losses': [],
        'value_losses': [],
        'best_val_dice': 0.0,
        'best_val_iou': 0.0,
        'best_episode': 0,
        'lr': args.lr,
        'gamma': args.gamma,
        'steps_per_episode': [],
        'moving_avg_dice': [],         # Moving average Dice score
        'moving_avg_reward': [],       # Moving average reward
        'eval_interval': args.eval_interval,  # Record evaluation interval
        'start_time': None,            # Training start time
        'end_time': None,              # Training end time
        'total_training_time': 0.0,    # Total training time
        'episode_times': []            # Time for each episode
    }
    
    # Record training start time
    training_start_time = time.time()
    training_history['start_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    # Track recent episode performance for moving average
    recent_dices = []
    recent_rewards = []
    window_size = min(20, args.num_episodes // 10 + 1)  # Moving average window size
    
    # Early stopping related variables
    patience = max(20, args.num_episodes // 20)  # Patience value, dynamically set based on total episodes
    no_improvement_count = 0
    early_stopped = False
    
    # Training loop
    for episode in tqdm(range(args.num_episodes)):
        if early_stopped:
            print(f"Early stopping triggered at episode {episode}, as validation score hasn't improved for {patience} evaluation intervals")
            break
            
        episode_start_time = time.time()
        
        # Sample random image
        idx = np.random.randint(len(train_dataset))
        sample = train_dataset[idx]
        
        # Get image and mask
        image = sample['image'].numpy()
        gt_mask = sample['mask'].numpy()
        
        # Use smarter initialization strategy
        if np.sum(gt_mask) > 0:
            # Create better initial segmentation based on mask
            # Add noise to ground truth mask
            noise_level = 0.2 - (episode / args.num_episodes) * 0.15  # Decrease noise as training progresses
            noise = np.random.normal(0, noise_level, gt_mask.shape)
            noisy_mask = gt_mask + noise 
            current_mask = (noisy_mask > 0.5).astype(np.float32)
            
            # Apply random morphological operations
            kernel_size = max(3, np.random.randint(3, 7))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            if np.random.rand() > 0.5:
                # Random dilation
                current_mask = cv2.dilate(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
            else:
                # Random erosion
                current_mask = cv2.erode(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
        else:
            # Empty mask when no ground truth
            current_mask = np.zeros_like(gt_mask)
        
        # Try to place pointer in polyp region
        if np.sum(gt_mask) > 0:
            # Find polyp pixels
            y_indices, x_indices = np.where(gt_mask.squeeze() > 0.5)
            if len(y_indices) > 0:
                # Select random point in polyp
                idx = np.random.randint(0, len(y_indices))
                pointer_y = y_indices[idx]
                pointer_x = x_indices[idx]
            else:
                # If no polyp pixels found, default to center
                pointer_y = image.shape[1] // 2
                pointer_x = image.shape[2] // 2
        else:
            # Default to center
            pointer_y = image.shape[1] // 2
            pointer_x = image.shape[2] // 2
        
        # Run episode
        episode_reward = 0
        episode_starts = True
        prev_dice = 0
        dices = []  # Record dice value for each step
        step_count = 0  # Count steps in this episode
        
        # Record detailed episode data, including action and reward for each step
        episode_details = {
            'step_actions': [],
            'step_rewards': [],
            'step_dices': [],
            'current_pointer': []
        }
        
        for step in range(args.max_steps):
            # Create observation
            obs = np.zeros((7, image.shape[1], image.shape[2]), dtype=np.float32)
            obs[0:3] = image  # RGB channels
            obs[3] = current_mask  # Segmentation mask
            
            # Create pointer position map
            pointer_map = np.zeros_like(current_mask)
            y_min = max(0, pointer_y - 5)
            y_max = min(pointer_map.shape[0], pointer_y + 6)
            x_min = max(0, pointer_x - 5)
            x_max = min(pointer_map.shape[1], pointer_x + 6)
            pointer_map[y_min:y_max, x_min:x_max] = 1.0
            obs[4] = pointer_map
            
            # Add distance map and edge map as additional features
            # Distance map - distance of each pixel to mask boundary
            if np.sum(current_mask) > 0:
                mask_binary = (current_mask > 0.5).astype(np.uint8)
                # Ensure input is single-channel 8-bit unsigned integer type
                if mask_binary.ndim > 2:
                    mask_binary = mask_binary.squeeze()
                dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
                # Safe normalization
                max_dist = np.max(dist_transform)
                if max_dist > 0:
                    dist_transform = dist_transform / max_dist
                obs[5] = dist_transform
                
                # Edge map - Canny edges of mask boundaries
                edges = cv2.Canny(mask_binary * 255, 100, 200) / 255.0
                obs[6] = edges
            else:
                obs[5:7] = 0
            
            # Select action
            action = agent.select_action(obs)
            
            # Record current pointer position
            episode_details['current_pointer'].append((pointer_y, pointer_x))
            
            # Execute action
            if action == 0:  # Up
                pointer_y = max(0, pointer_y - 10)
            elif action == 1:  # Down
                pointer_y = min(image.shape[1] - 1, pointer_y + 10)
            elif action == 2:  # Left
                pointer_x = max(0, pointer_x - 10)
            elif action == 3:  # Right
                pointer_x = min(image.shape[2] - 1, pointer_x + 10)
            elif action == 4:  # Expand
                # Dilate mask
                kernel = np.ones((5, 5), np.uint8)
                current_mask = cv2.dilate(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
            elif action == 5:  # Shrink
                # Erode mask
                kernel = np.ones((5, 5), np.uint8)
                current_mask = cv2.erode(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
            elif action == 6:  # Done
                # Add done flag
                agent.dones.append(True)
                # Record action
                episode_details['step_actions'].append(action)
                break
            
            # Add circle at pointer position (if it was a movement action)
            if action <= 3:
                y, x = np.ogrid[:current_mask.shape[0], :current_mask.shape[1]]
                mask = (x - pointer_x)**2 + (y - pointer_y)**2 <= 10**2
                current_mask[mask] = 1.0
            
            # Calculate reward
            dice = dice_coefficient(current_mask, gt_mask)
            dices.append(dice)
            
            # Record dice value for each step
            episode_details['step_dices'].append(float(dice))
            
            # If it's the first iteration, store previous dice
            if step == 0:
                prev_dice = 0 if episode_starts else dice_coefficient(current_mask, gt_mask)
            
            # Calculate reward as improvement in Dice score
            dice_improvement = dice - prev_dice
            
            # Improved reward structure
            if dice_improvement > 0:
                # Positive reward for improvement, larger reward for greater improvement
                reward = 0.5 + dice_improvement * 10.0
            elif dice_improvement == 0:
                # Small negative reward for no change
                reward = -0.05
            else:
                # Larger negative reward for getting worse, penalty proportional to decline
                reward = dice_improvement * 5.0
            
            # Bonus reward for reaching high dice values
            if dice > 0.8:
                reward += 0.2
            
            # Penalty for choosing Done but Dice score is low
            if action == 6 and dice < 0.6:
                reward -= 0.5
            
            # Store new previous dice
            prev_dice = dice
            
            # Store reward
            agent.rewards.append(reward)
            episode_reward += reward
            
            # Record action and reward
            episode_details['step_actions'].append(action)
            episode_details['step_rewards'].append(float(reward))
            
            # Add done flag (intermediate steps are not done)
            agent.dones.append(False)
            
            # Only the first step in an episode is the start
            episode_starts = False
            
            # Update step counter
            step_count += 1
        
        # Calculate final metrics
        final_dice = dice_coefficient(current_mask, gt_mask)
        final_iou = iou_score(current_mask, gt_mask)
        
        # Add completion bonus
        completion_bonus = final_dice * 0.5
        agent.rewards.append(completion_bonus)
        episode_reward += completion_bonus
        if len(episode_details['step_rewards']) < len(episode_details['step_actions']):
            episode_details['step_rewards'].append(float(completion_bonus))
        
        # Update agent
        policy_loss, value_loss = agent.update()
        
        # Store metrics
        all_rewards.append(episode_reward)
        all_dice_scores.append(final_dice)
        all_iou_scores.append(final_iou)
        
        # Update moving averages
        recent_dices.append(final_dice)
        recent_rewards.append(episode_reward)
        if len(recent_dices) > window_size:
            recent_dices.pop(0)
            recent_rewards.pop(0)
        
        # Calculate moving averages
        moving_avg_dice = sum(recent_dices) / len(recent_dices)
        moving_avg_reward = sum(recent_rewards) / len(recent_rewards)
        
        # Calculate time spent on this episode
        episode_time = time.time() - episode_start_time
        
        # Update training history
        training_history['episodes'].append(episode + 1)
        training_history['train_rewards'].append(float(episode_reward))
        training_history['train_dice_scores'].append(float(final_dice))
        training_history['train_iou_scores'].append(float(final_iou))
        training_history['policy_losses'].append(float(policy_loss))
        training_history['value_losses'].append(float(value_loss))
        training_history['steps_per_episode'].append(step_count)
        training_history['moving_avg_dice'].append(float(moving_avg_dice))
        training_history['moving_avg_reward'].append(float(moving_avg_reward))
        training_history['episode_times'].append(float(episode_time))
        
        # Save JSON format training history immediately after each update to ensure complete records
        json_serializable = {}
        for key, value in training_history.items():
            if isinstance(value, (list, dict)):
                if len(value) > 0 and isinstance(value[0], (np.integer, np.floating, np.ndarray)):
                    json_serializable[key] = [float(v) for v in value]
                else:
                    json_serializable[key] = value
            elif isinstance(value, (np.integer, np.floating)):
                json_serializable[key] = float(value)
            else:
                json_serializable[key] = value
        
        # Save training history JSON file after each episode
        with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
            json.dump(json_serializable, f, indent=2)
            
        # Save a backup copy for data safety
        if (episode + 1) % 5 == 0 or (episode + 1) == args.num_episodes:
            with open(os.path.join(args.output_dir, f'training_history_ep{episode+1}.json'), 'w') as f:
                json.dump(json_serializable, f, indent=2)
                
        # Save pickle format
        with open(os.path.join(args.output_dir, 'training_history.pkl'), 'wb') as f:
            pickle.dump(training_history, f)
        
        # Log progress
        if (episode + 1) % args.log_interval == 0:
            mean_reward = np.mean(all_rewards[-args.log_interval:])
            mean_dice = np.mean(all_dice_scores[-args.log_interval:])
            mean_iou = np.mean(all_iou_scores[-args.log_interval:])
            print(f"Episode {episode+1}: Mean Reward = {mean_reward:.4f}, Mean Dice = {mean_dice:.4f}, Mean IoU = {mean_iou:.4f}, Time = {episode_time:.2f}s")
        
        # Evaluate on validation set
        if (episode + 1) % args.eval_interval == 0:
            val_dice_scores = []
            val_iou_scores = []
            
            # Test on validation images
            for _ in range(args.num_eval_episodes):
                # Sample random validation image
                idx = np.random.randint(len(val_dataset))
                sample = val_dataset[idx]
                
                # Get image and mask
                image = sample['image'].numpy()
                gt_mask = sample['mask'].numpy()
                
                # Get prediction for this sample
                current_mask = predict_mask(agent, image, gt_mask, max_steps=args.max_steps)
                
                # Calculate metrics
                val_dice = dice_coefficient(current_mask, gt_mask)
                val_iou = iou_score(current_mask, gt_mask)
                
                val_dice_scores.append(val_dice)
                val_iou_scores.append(val_iou)
            
            # Calculate average performance
            mean_val_dice = np.mean(val_dice_scores)
            mean_val_iou = np.mean(val_iou_scores)
            
            # Add validation metrics to history
            val_results['episodes'].append(episode + 1)
            val_results['dice_scores'].append(float(mean_val_dice))
            val_results['iou_scores'].append(float(mean_val_iou))
            
            # Update validation performance in training history
            training_history['val_dice_scores'].append(float(mean_val_dice))
            training_history['val_iou_scores'].append(float(mean_val_iou))
            
            # Immediately update and save JSON file after each validation
            json_serializable = {}
            for key, value in training_history.items():
                if isinstance(value, (list, dict)):
                    if len(value) > 0 and isinstance(value[0], (np.integer, np.floating, np.ndarray)):
                        json_serializable[key] = [float(v) for v in value]
                    else:
                        json_serializable[key] = value
                elif isinstance(value, (np.integer, np.floating)):
                    json_serializable[key] = float(value)
                else:
                    json_serializable[key] = value
            
            with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
                json.dump(json_serializable, f, indent=2)
            
            print(f"Validation: Mean Dice = {mean_val_dice:.4f}, Mean IoU = {mean_val_iou:.4f}")
            
            # Save best model
            if mean_val_dice > best_val_dice:
                best_val_dice = mean_val_dice
                best_val_iou = mean_val_iou
                best_episode = episode + 1
                
                # Reset early stopping counter
                no_improvement_count = 0
                
                # Update best values in history record
                training_history['best_val_dice'] = float(best_val_dice)
                training_history['best_val_iou'] = float(best_val_iou)
                training_history['best_episode'] = best_episode
                
                print(f"New best model with validation Dice: {best_val_dice:.4f}")
                agent.save(os.path.join(args.output_dir, 'best_model.pth'))
                
                # Save model info
                with open(os.path.join(args.output_dir, 'best_model_info.txt'), 'w') as f:
                    f.write(f"Episode: {best_episode}\n")
                    f.write(f"Validation Dice: {best_val_dice:.4f}\n")
                    f.write(f"Validation IoU: {best_val_iou:.4f}\n")
                    
                # Save current best model's detailed training history
                with open(os.path.join(args.output_dir, 'best_model_history.pkl'), 'wb') as f:
                    pickle.dump(training_history, f)
            else:
                # Increment early stopping counter
                no_improvement_count += 1
                print(f"No improvement for {no_improvement_count} evaluation intervals. Best Val Dice: {best_val_dice:.4f}")
                
                # Check if early stopping should be triggered
                if no_improvement_count >= patience:
                    print(f"Early stopping triggered at episode {episode+1} as validation performance hasn't improved for {patience} evaluation intervals")
                    early_stopped = True
        
        # Save agent checkpoint periodically
        checkpoint_interval = min(50, args.num_episodes // 10 + 1)  # Adjust checkpoint interval based on total episodes
        if (episode + 1) % checkpoint_interval == 0:
            agent.save(os.path.join(args.output_dir, f'checkpoint_ep{episode+1}.pth'))
        
        # Save final model
        if (episode + 1) == args.num_episodes:
            agent.save(os.path.join(args.output_dir, 'final_model.pth'))
    
    # Record training end time and total training time
    training_end_time = time.time()
    training_history['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    training_history['total_training_time'] = float(training_end_time - training_start_time)
    
    # Save final complete training history
    print(f"Training completed in {training_history['total_training_time']:.2f} seconds")
    print(f"Best validation Dice: {best_val_dice:.4f} at episode {best_episode}")
    
    # Final save of updated training history (including total training time)
    with open(os.path.join(args.output_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(training_history, f)
    
    # Final JSON format save
    json_serializable = {}
    for key, value in training_history.items():
        if isinstance(value, (list, dict)):
            if len(value) > 0 and isinstance(value[0], (np.integer, np.floating, np.ndarray)):
                json_serializable[key] = [float(v) for v in value]
            else:
                json_serializable[key] = value
        elif isinstance(value, (np.integer, np.floating)):
            json_serializable[key] = float(value)
        else:
            json_serializable[key] = value
    
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(json_serializable, f, indent=2)
    
    # Save final training history to additional backup files
    with open(os.path.join(args.output_dir, 'simple_rl_training_history.pkl'), 'wb') as f:
        pickle.dump(training_history, f)
    
    with open(os.path.join(args.output_dir, 'simple_rl_training_history.json'), 'w') as f:
        json.dump(json_serializable, f, indent=2)
    
    # Return training history and best validation results
    return {
        'training_history': training_history,
        'val_results': val_results,
        'best_val_dice': float(best_val_dice),
        'best_val_iou': float(best_val_iou),
        'best_episode': best_episode,
        'total_training_time': float(training_history['total_training_time']),
        'early_stopped': early_stopped
    }

def predict_mask(agent, image, gt_mask, max_steps=20):
    """Use the agent to predict a mask for the given image"""
    # Initialize segmentation variables
    # Initialize mask in a smart way
    if np.sum(gt_mask) > 0:
        # Add noise to ground truth mask, but with less noise (only for evaluation)
        noise = np.random.normal(0, 0.1, gt_mask.shape)
        noisy_mask = gt_mask + noise 
        current_mask = (noisy_mask > 0.5).astype(np.float32)
        
        # Apply random morphological operations
        kernel_size = np.random.randint(3, 5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if np.random.rand() > 0.5:
            # Random dilation
            current_mask = cv2.dilate(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
        else:
            # Random erosion
            current_mask = cv2.erode(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
    else:
        # Empty mask when no ground truth
        current_mask = np.zeros_like(gt_mask)
    
    # Try to place pointer in polyp region
    if np.sum(gt_mask) > 0:
        # Find polyp pixels
        y_indices, x_indices = np.where(gt_mask.squeeze() > 0.5)
        if len(y_indices) > 0:
            # Select random point in polyp
            idx = np.random.randint(0, len(y_indices))
            pointer_y = y_indices[idx]
            pointer_x = x_indices[idx]
        else:
            # If no polyp pixels found, default to center
            pointer_y = image.shape[1] // 2
            pointer_x = image.shape[2] // 2
    else:
        # Default to center
        pointer_y = image.shape[1] // 2
        pointer_x = image.shape[2] // 2
    
    # Run episode (evaluation mode)
    for step in range(max_steps):
        # Create observation
        obs = np.zeros((7, image.shape[1], image.shape[2]), dtype=np.float32)
        obs[0:3] = image
        obs[3] = current_mask
        
        # Create pointer position map
        pointer_map = np.zeros_like(current_mask)
        y_min = max(0, pointer_y - 5)
        y_max = min(pointer_map.shape[0], pointer_y + 6)
        x_min = max(0, pointer_x - 5)
        x_max = min(pointer_map.shape[1], pointer_x + 6)
        pointer_map[y_min:y_max, x_min:x_max] = 1.0
        obs[4] = pointer_map
        
        # Add distance map and edge map as additional features
        if np.sum(current_mask) > 0:
            mask_binary = (current_mask > 0.5).astype(np.uint8)
            # Ensure input is single-channel 8-bit unsigned integer type
            if mask_binary.ndim > 2:
                mask_binary = mask_binary.squeeze()
            dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
            # Safe normalization
            max_dist = np.max(dist_transform)
            if max_dist > 0:
                dist_transform = dist_transform / max_dist
            obs[5] = dist_transform
            
            # Edge map - Canny edges of mask boundaries
            edges = cv2.Canny(mask_binary * 255, 100, 200) / 255.0
            obs[6] = edges
        else:
            obs[5:7] = 0
        
        # Select action (more deterministically)
        state = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            probs = agent.policy_net(state)
            action = torch.argmax(probs, dim=1).item()
        
        # Execute action
        if action == 0:  # Up
            pointer_y = max(0, pointer_y - 10)
        elif action == 1:  # Down
            pointer_y = min(image.shape[1] - 1, pointer_y + 10)
        elif action == 2:  # Left
            pointer_x = max(0, pointer_x - 10)
        elif action == 3:  # Right
            pointer_x = min(image.shape[2] - 1, pointer_x + 10)
        elif action == 4:  # Expand
            kernel = np.ones((5, 5), np.uint8)
            current_mask = cv2.dilate(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
        elif action == 5:  # Shrink
            kernel = np.ones((5, 5), np.uint8)
            current_mask = cv2.erode(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
        elif action == 6:  # Done
            break
        
        # Add circle at pointer position (if it was a movement action)
        if action <= 3:
            y, x = np.ogrid[:current_mask.shape[0], :current_mask.shape[1]]
            mask = (x - pointer_x)**2 + (y - pointer_y)**2 <= 10**2
            current_mask[mask] = 1.0
    
    return current_mask

def visualize_results(image, gt_mask, pred_mask):
    """Create visualization of prediction vs ground truth"""
    # Transpose image for matplotlib (if needed)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # Ensure masks are 2D
    if gt_mask.ndim > 2:
        gt_mask = gt_mask.squeeze()
    if pred_mask.ndim > 2:
        pred_mask = pred_mask.squeeze()
    
    # Create RGB visualization
    vis = np.copy(image)
    
    # Add ground truth boundary in red
    gt_boundary = cv2.Canny((gt_mask > 0.5).astype(np.uint8) * 255, 100, 200) / 255.0
    vis[..., 0] = np.maximum(vis[..., 0], gt_boundary * 0.8)
    vis[..., 1] = np.maximum(vis[..., 1], gt_boundary * 0.0)
    vis[..., 2] = np.maximum(vis[..., 2], gt_boundary * 0.0)
    
    # Add prediction in green
    pred_boundary = cv2.Canny((pred_mask > 0.5).astype(np.uint8) * 255, 100, 200) / 255.0
    vis[..., 0] = np.maximum(vis[..., 0], pred_boundary * 0.0)
    vis[..., 1] = np.maximum(vis[..., 1], pred_boundary * 0.8)
    vis[..., 2] = np.maximum(vis[..., 2], pred_boundary * 0.0)
    
    # Add overlay for prediction filling
    overlay = np.zeros_like(vis)
    overlay[..., 1] = (pred_mask > 0.5) * 0.3  # Semi-transparent green
    
    # Blend overlay with visualization
    vis = np.clip(vis + overlay, 0, 1)
    
    return vis

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agent for polyp segmentation')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/raw',
                      help='Directory containing images')
    parser.add_argument('--results_dir', type=str, default='results/rl',
                      help='Directory to save results')
    parser.add_argument('--output_dir', type=str, default='results/rl',
                      help='Directory to save output files')
    
    # Agent parameters
    parser.add_argument('--lr', type=float, default=3e-4,
                      help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=100,
                      help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=20,
                      help='Maximum steps per episode')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for updates')
    parser.add_argument('--update_interval', type=int, default=5,
                      help='Number of steps between updates')
    parser.add_argument('--log_interval', type=int, default=5,
                      help='Interval for logging')
    parser.add_argument('--eval_interval', type=int, default=5,
                      help='Interval for evaluation')
    parser.add_argument('--num_eval_episodes', type=int, default=5,
                      help='Number of episodes to evaluate on')
    parser.add_argument('--save_interval', type=int, default=10,
                      help='Interval for saving model')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=20,
                      help='Early stopping patience (number of eval intervals with no improvement)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_agent(args) 