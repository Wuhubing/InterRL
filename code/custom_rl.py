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
from collections import deque, namedtuple

from code.data_utils import PolypDataset
from code.utils import dice_coefficient, iou_score, set_seed

# Experience tuple for storing transitions
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class SimpleCNN(nn.Module):
    """Simple CNN for image features"""
    def __init__(self, in_channels=7):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # Linear layer size will be computed dynamically
        self.fc_size = None
        self.fc = None
        
    def _init_fc(self, x_shape):
        """Initialize the fully connected layer with the correct input size"""
        # Compute the size after convolutions
        with torch.no_grad():
            # Pass a dummy tensor through convolutions to get shape
            dummy = torch.zeros(1, x_shape[0], x_shape[1], x_shape[2])
            out = self.conv2(self.conv1(dummy))
            fc_in_features = out.view(1, -1).shape[1]
        
        # Initialize the fully connected layer
        self.fc = nn.Linear(fc_in_features, 256)
        self.fc_size = fc_in_features
        
    def forward(self, x):
        # Initialize FC layer if needed
        if self.fc is None:
            self._init_fc(x.shape)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class PolicyNetwork(nn.Module):
    """Policy network for the agent"""
    def __init__(self, in_channels=7, num_actions=7):
        super(PolicyNetwork, self).__init__()
        self.cnn = SimpleCNN(in_channels)
        self.action_head = nn.Linear(256, num_actions)
        
    def forward(self, x):
        x = self.cnn(x)
        action_probs = F.softmax(self.action_head(x), dim=-1)
        return action_probs

class ValueNetwork(nn.Module):
    """Value network for the agent"""
    def __init__(self, in_channels=7):
        super(ValueNetwork, self).__init__()
        self.cnn = SimpleCNN(in_channels)
        self.value_head = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.cnn(x)
        value = self.value_head(x)
        return value

class RLAgent:
    """Reinforcement learning agent using simple Actor-Critic architecture"""
    def __init__(self, in_channels=7, num_actions=7, lr=3e-4, gamma=0.99, device='cpu'):
        self.device = torch.device(device)
        self.policy_net = PolicyNetwork(in_channels, num_actions).to(self.device)
        self.value_net = ValueNetwork(in_channels).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.experience_buffer = []
        
    def select_action(self, state):
        """Select action based on current state"""
        state = state.to(self.device)
        with torch.no_grad():
            action_probs = self.policy_net(state.unsqueeze(0))
            action = torch.multinomial(action_probs, 1).item()
        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in buffer"""
        self.experience_buffer.append(Experience(state, action, reward, next_state, done))
    
    def update(self, batch_size=16):
        """Update policy and value networks"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample random batch from buffer
        batch = random.sample(self.experience_buffer, batch_size)
        
        # Convert to tensors
        states = torch.stack([exp.state for exp in batch])
        actions = torch.tensor([exp.action for exp in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float).to(self.device)
        next_states = torch.stack([exp.next_state for exp in batch])
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.float).to(self.device)
        
        # Calculate returns
        with torch.no_grad():
            next_values = self.value_net(next_states).squeeze()
            returns = rewards + self.gamma * next_values * (1 - dones)
        
        # Get current values and action probs
        values = self.value_net(states).squeeze()
        action_probs = self.policy_net(states)
        
        # Calculate advantages
        advantages = returns - values
        
        # Calculate losses
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns.detach())
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def save(self, path):
        """Save agent to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """Load agent from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agent for polyp segmentation')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/raw',
                      help='Directory containing images')
    parser.add_argument('--results_dir', type=str, default='results/rl',
                      help='Directory to save results')
    
    # Agent parameters
    parser.add_argument('--lr', type=float, default=3e-4,
                      help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=50,
                      help='Number of episodes to train for')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for updates')
    parser.add_argument('--update_interval', type=int, default=5,
                      help='Number of steps between updates')
    parser.add_argument('--eval_interval', type=int, default=5,
                      help='Interval for evaluation')
    parser.add_argument('--save_interval', type=int, default=10,
                      help='Interval for saving model')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    return parser.parse_args()

def load_dataset(args):
    """Load dataset"""
    print(f"Loading dataset from {args.data_dir}")
    train_dataset = PolypDataset(args.data_dir, split='train')
    val_dataset = PolypDataset(args.data_dir, split='val')
    
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    return train_dataset, val_dataset

def simulate_episode(agent, image, gt_mask, max_steps=20):
    """Simulate a full episode"""
    # Initialize environment state
    current_mask = np.zeros_like(gt_mask)
    pointer_x = image.shape[2] // 2
    pointer_y = image.shape[1] // 2
    
    # Initialize observation (7 channels)
    obs = np.zeros((7, image.shape[1], image.shape[2]), dtype=np.float32)
    obs[0:3] = image  # RGB image
    obs[3] = current_mask  # Current mask
    
    # Create initial pointer position
    pointer_map = np.zeros_like(current_mask)
    y_min = max(0, pointer_y - 5)
    y_max = min(pointer_map.shape[0], pointer_y + 6)
    x_min = max(0, pointer_x - 5)
    x_max = min(pointer_map.shape[1], pointer_x + 6)
    pointer_map[y_min:y_max, x_min:x_max] = 1.0
    
    obs[4] = pointer_map  # Pointer position
    obs[5] = np.zeros_like(current_mask)  # Distance map (placeholder)
    obs[6] = np.zeros_like(current_mask)  # Edge map (placeholder)
    
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(agent.device)
    
    # Episode variables
    done = False
    step = 0
    episode_reward = 0
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    
    # Run episode
    while not done and step < max_steps:
        # Store current state
        states.append(obs_tensor)
        
        # Select action
        action = agent.select_action(obs_tensor)
        actions.append(action)
        
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
            # Simulate expansion with a simple dilation
            import cv2
            kernel = np.ones((3, 3), np.uint8)
            current_mask = cv2.dilate(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
        elif action == 5:  # Shrink
            # Simulate shrinkage with a simple erosion
            import cv2
            kernel = np.ones((3, 3), np.uint8)
            current_mask = cv2.erode(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
        elif action == 6:  # Confirm (done)
            done = True
        
        # Update mask at pointer location if not done
        if not done and action < 4:  # Only for movement actions
            # Add a circle at pointer location
            y_coords, x_coords = np.ogrid[:current_mask.shape[0], :current_mask.shape[1]]
            circle_mask = ((y_coords - pointer_y)**2 + (x_coords - pointer_x)**2 <= 10**2)
            current_mask[circle_mask] = 1.0
        
        # Calculate reward (improvement in Dice coefficient)
        dice = dice_coefficient(current_mask, gt_mask)
        
        # Basic reward structure
        reward = 0.1 * dice  # Proportional to current dice
        
        if done:
            reward += dice  # Bonus for finishing
        
        rewards.append(reward)
        
        # Update observation
        next_obs = np.zeros_like(obs)
        next_obs[0:3] = image  # RGB image
        next_obs[3] = current_mask  # Updated mask
        
        # Update pointer position
        pointer_map = np.zeros_like(current_mask)
        y_min = max(0, pointer_y - 5)
        y_max = min(pointer_map.shape[0], pointer_y + 6)
        x_min = max(0, pointer_x - 5)
        x_max = min(pointer_map.shape[1], pointer_x + 6)
        pointer_map[y_min:y_max, x_min:x_max] = 1.0
        
        next_obs[4] = pointer_map
        next_obs[5] = np.zeros_like(current_mask)  # Simple placeholder
        next_obs[6] = np.zeros_like(current_mask)  # Simple placeholder
        
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(agent.device)
        next_states.append(next_obs_tensor)
        
        dones.append(done)
        
        # Update state and counters
        obs = next_obs
        obs_tensor = next_obs_tensor
        episode_reward += reward
        step += 1
    
    # Store experiences in buffer
    for i in range(len(states)):
        agent.store_experience(
            states[i], 
            actions[i], 
            rewards[i], 
            next_states[i], 
            dones[i]
        )
    
    # Calculate final metrics
    final_dice = dice_coefficient(current_mask, gt_mask)
    final_iou = iou_score(current_mask, gt_mask)
    
    return episode_reward, final_dice, final_iou, step

def train(args):
    """Train RL agent"""
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load dataset
    train_dataset, val_dataset = load_dataset(args)
    
    # Create agent
    agent = RLAgent(
        in_channels=7,
        num_actions=7,
        lr=args.lr,
        gamma=args.gamma,
        device=args.device
    )
    
    # Training metrics
    episode_rewards = []
    episode_dices = []
    episode_ious = []
    
    best_dice = 0.0
    
    # Training loop
    print("Starting training...")
    pbar = tqdm(range(args.num_episodes))
    for episode in pbar:
        # Sample a random image from the dataset
        idx = random.randint(0, len(train_dataset) - 1)
        sample = train_dataset[idx]
        
        # Get image and ground truth mask
        image = sample['image'].numpy()
        gt_mask = sample['mask'].numpy()
        
        # Simulate episode
        reward, dice, iou, steps = simulate_episode(agent, image, gt_mask)
        
        # Store metrics
        episode_rewards.append(reward)
        episode_dices.append(dice)
        episode_ious.append(iou)
        
        # Update agent
        if len(agent.experience_buffer) >= args.batch_size:
            loss_info = agent.update(batch_size=args.batch_size)
            
            # Update progress bar
            pbar.set_description(
                f"Episode {episode+1}/{args.num_episodes} | "
                f"Reward: {reward:.2f} | "
                f"Dice: {dice:.2f} | "
                f"IoU: {iou:.2f} | "
                f"PL: {loss_info['policy_loss']:.4f} | "
                f"VL: {loss_info['value_loss']:.4f}"
            )
        else:
            pbar.set_description(
                f"Episode {episode+1}/{args.num_episodes} | "
                f"Reward: {reward:.2f} | "
                f"Dice: {dice:.2f} | "
                f"IoU: {iou:.2f}"
            )
        
        # Evaluation
        if (episode + 1) % args.eval_interval == 0:
            val_rewards = []
            val_dices = []
            val_ious = []
            
            # Evaluate on 5 random validation samples
            for _ in range(5):
                idx = random.randint(0, len(val_dataset) - 1)
                sample = val_dataset[idx]
                
                image = sample['image'].numpy()
                gt_mask = sample['mask'].numpy()
                
                reward, dice, iou, _ = simulate_episode(agent, image, gt_mask)
                
                val_rewards.append(reward)
                val_dices.append(dice)
                val_ious.append(iou)
            
            # Calculate average metrics
            avg_val_reward = np.mean(val_rewards)
            avg_val_dice = np.mean(val_dices)
            avg_val_iou = np.mean(val_ious)
            
            print(f"\nValidation: Reward: {avg_val_reward:.4f}, Dice: {avg_val_dice:.4f}, IoU: {avg_val_iou:.4f}")
            
            # Save best model
            if avg_val_dice > best_dice:
                best_dice = avg_val_dice
                agent.save(os.path.join(args.results_dir, 'best_agent.pt'))
                print(f"New best model saved with Dice score: {best_dice:.4f}")
        
        # Save model
        if (episode + 1) % args.save_interval == 0:
            agent.save(os.path.join(args.results_dir, f'agent_episode_{episode+1}.pt'))
    
    # Save final model
    agent.save(os.path.join(args.results_dir, 'final_agent.pt'))
    
    # Plot metrics
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(episode_dices)
    plt.title('Dice Scores')
    plt.xlabel('Episode')
    plt.ylabel('Dice')
    
    plt.subplot(1, 3, 3)
    plt.plot(episode_ious)
    plt.title('IoU Scores')
    plt.xlabel('Episode')
    plt.ylabel('IoU')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'training_metrics.png'))
    plt.close()
    
    print(f"Training completed. Best Dice score: {best_dice:.4f}")

if __name__ == '__main__':
    args = parse_args()
    train(args) 