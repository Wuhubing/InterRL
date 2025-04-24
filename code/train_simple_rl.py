import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
from collections import deque

from code.data_utils import PolypDataset
from code.enhanced_rl_agent import EnhancedPPOAgent
from code.utils import dice_coefficient, iou_score, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agent for polyp segmentation')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/raw',
                      help='Directory containing images')
    parser.add_argument('--results_dir', type=str, default='results/rl',
                      help='Directory to save results')
    
    # Agent parameters
    parser.add_argument('--hidden_size', type=int, default=128,
                      help='Hidden size of the agent')
    parser.add_argument('--lr', type=float, default=3e-4,
                      help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=100,
                      help='Number of episodes to train for')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for updates')
    parser.add_argument('--update_interval', type=int, default=10,
                      help='Number of steps between updates')
    parser.add_argument('--eval_interval', type=int, default=10,
                      help='Interval for evaluation')
    parser.add_argument('--save_interval', type=int, default=20,
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

def create_agent(args, obs_shape, action_space):
    """Create agent"""
    device = torch.device(args.device)
    
    agent = EnhancedPPOAgent(
        obs_shape=obs_shape,
        action_space=action_space,
        hidden_size=args.hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        device=device
    )
    
    return agent

def train(args):
    """Train agent"""
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load dataset
    train_dataset, val_dataset = load_dataset(args)
    
    # Define action space and observation space
    action_space = 7  # Up, Down, Left, Right, Expand, Shrink, Confirm
    
    # Get a sample from the dataset to determine observation shape
    sample = train_dataset[0]
    image = sample['image'].numpy()
    mask = sample['mask'].numpy()
    
    # Define observation shape: image (3 channels) + current mask (1 channel) + previous action masks (3 channels)
    obs_shape = (7, image.shape[1], image.shape[2])
    
    # Create agent
    agent = create_agent(args, obs_shape, action_space)
    
    # Training metrics
    episode_rewards = []
    episode_dice_scores = []
    episode_iou_scores = []
    
    best_dice = 0.0
    
    # Training loop
    print("Starting training...")
    for episode in range(args.num_episodes):
        # Sample a random image from the dataset
        idx = random.randint(0, len(train_dataset) - 1)
        sample = train_dataset[idx]
        
        # Get image and ground truth mask
        image = sample['image'].numpy()
        gt_mask = sample['mask'].numpy()
        
        # Initialize environment state
        current_mask = np.zeros_like(gt_mask)
        pointer_x = image.shape[2] // 2
        pointer_y = image.shape[1] // 2
        
        # Initialize observation
        obs = np.zeros(obs_shape, dtype=np.float32)
        obs[0:3] = image  # RGB image
        obs[3] = current_mask  # Current mask
        obs[4] = np.zeros_like(current_mask)  # Pointer position
        obs[5] = np.zeros_like(current_mask)  # Distance map
        obs[6] = np.zeros_like(current_mask)  # Edge map
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(agent.device)
        
        # Episode variables
        done = False
        step = 0
        episode_reward = 0
        
        # Run episode
        while not done and step < 20:  # Maximum 20 steps per episode
            # Select action
            action = agent.select_action(obs_tensor)
            
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
                kernel = np.ones((3, 3), np.uint8)
                current_mask = cv2.dilate(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
            elif action == 5:  # Shrink
                # Simulate shrinkage with a simple erosion
                kernel = np.ones((3, 3), np.uint8)
                current_mask = cv2.erode(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
            elif action == 6:  # Confirm (done)
                done = True
            
            # Update mask at pointer location if not done
            if not done:
                # Add a circle at pointer location
                y, x = np.ogrid[-10:11, -10:11]
                mask = x**2 + y**2 <= 10**2
                
                y_min = max(0, pointer_y - 10)
                y_max = min(image.shape[1], pointer_y + 11)
                x_min = max(0, pointer_x - 10)
                x_max = min(image.shape[2], pointer_x + 11)
                
                mask_y, mask_x = np.where(mask)
                mask_y = mask_y + (y_min - (pointer_y - 10))
                mask_x = mask_x + (x_min - (pointer_x - 10))
                
                valid_indices = (
                    (mask_y >= 0) & 
                    (mask_y < current_mask.shape[0]) & 
                    (mask_x >= 0) & 
                    (mask_x < current_mask.shape[1])
                )
                
                mask_y = mask_y[valid_indices]
                mask_x = mask_x[valid_indices]
                
                current_mask[mask_y, mask_x] = 1.0
            
            # Calculate reward (improvement in Dice coefficient)
            dice = dice_coefficient(current_mask, gt_mask)
            iou = iou_score(current_mask, gt_mask)
            
            # Calculate reward as improvement in Dice coefficient
            reward = dice * 0.1  # Small positive reward proportional to dice
            
            if done:
                reward += dice  # Bonus reward at the end
            
            # Update observation
            next_obs = np.zeros_like(obs)
            next_obs[0:3] = image  # RGB image
            next_obs[3] = current_mask  # Updated mask
            
            # Create pointer position channel
            pointer_channel = np.zeros_like(current_mask)
            
            # Create a simpler pointer marker
            y_min = max(0, pointer_y - 5)
            y_max = min(pointer_channel.shape[0], pointer_y + 6)
            x_min = max(0, pointer_x - 5)
            x_max = min(pointer_channel.shape[1], pointer_x + 6)
            
            # Set a simple square marker for pointer
            pointer_channel[y_min:y_max, x_min:x_max] = 1.0
            next_obs[4] = pointer_channel
            
            # Create distance map and edge map
            dist_map = np.zeros_like(current_mask)
            edge_map = np.zeros_like(current_mask)
            if current_mask.sum() > 0:
                # Simple distance transform
                from scipy.ndimage import distance_transform_edt
                dist_map = distance_transform_edt(current_mask.squeeze() > 0.5)
                if dist_map.max() > 0:
                    dist_map = dist_map / dist_map.max()
                
                # Simple edge detection
                import cv2
                binary_mask = (current_mask.squeeze() > 0.5).astype(np.uint8) * 255
                if binary_mask.ndim == 2:  # Ensure mask is 2D for Canny
                    edge_map = cv2.Canny(binary_mask, 100, 200) / 255.0
                else:
                    edge_map = np.zeros_like(binary_mask)
            
            # Ensure shapes match observation
            dist_map = dist_map.reshape(current_mask.shape)
            edge_map = edge_map.reshape(current_mask.shape)
            
            next_obs[5] = dist_map
            next_obs[6] = edge_map
            
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(agent.device)
            
            # Store experience
            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(agent.device)
            done_tensor = torch.tensor([float(done)], dtype=torch.float32).to(agent.device)
            
            agent.store_experience(
                obs=obs_tensor,
                action=action,
                reward=reward_tensor,
                next_obs=next_obs_tensor,
                done=done_tensor
            )
            
            # Update variables
            obs = next_obs
            obs_tensor = next_obs_tensor
            episode_reward += reward
            step += 1
            
            # Update agent
            if (episode * 20 + step) % args.update_interval == 0 and len(agent.experience_buffer) >= args.batch_size:
                agent.update(batch_size=args.batch_size)
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_dice_scores.append(dice)
        episode_iou_scores.append(iou)
        
        # Print metrics every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_dice = np.mean(episode_dice_scores[-10:])
            avg_iou = np.mean(episode_iou_scores[-10:])
            
            print(f"Episode {episode+1}/{args.num_episodes}: "
                  f"Reward: {avg_reward:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")
        
        # Evaluate model
        if (episode + 1) % args.eval_interval == 0:
            val_dice_scores = []
            val_iou_scores = []
            
            # Evaluate on validation set (using 5 samples)
            for _ in range(5):
                idx = random.randint(0, len(val_dataset) - 1)
                sample = val_dataset[idx]
                
                # Get image and ground truth mask
                image = sample['image'].numpy()
                gt_mask = sample['mask'].numpy()
                
                # Initialize environment state
                current_mask = np.zeros_like(gt_mask)
                pointer_x = image.shape[2] // 2
                pointer_y = image.shape[1] // 2
                
                # Initialize observation
                obs = np.zeros(obs_shape, dtype=np.float32)
                obs[0:3] = image  # RGB image
                obs[3] = current_mask  # Current mask
                obs[4] = np.zeros_like(current_mask)  # Pointer position
                obs[5] = np.zeros_like(current_mask)  # Distance map
                obs[6] = np.zeros_like(current_mask)  # Edge map
                
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(agent.device)
                
                # Run episode
                done = False
                step = 0
                
                while not done and step < 20:
                    # Select action
                    with torch.no_grad():
                        action = agent.select_action(obs_tensor)
                    
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
                        kernel = np.ones((3, 3), np.uint8)
                        current_mask = cv2.dilate(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
                    elif action == 5:  # Shrink
                        # Simulate shrinkage with a simple erosion
                        kernel = np.ones((3, 3), np.uint8)
                        current_mask = cv2.erode(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
                    elif action == 6:  # Confirm (done)
                        done = True
                    
                    # Update mask at pointer location if not done
                    if not done:
                        # Add a circle at pointer location
                        y, x = np.ogrid[-10:11, -10:11]
                        mask = x**2 + y**2 <= 10**2
                        
                        y_min = max(0, pointer_y - 10)
                        y_max = min(image.shape[1], pointer_y + 11)
                        x_min = max(0, pointer_x - 10)
                        x_max = min(image.shape[2], pointer_x + 11)
                        
                        mask_y, mask_x = np.where(mask)
                        mask_y = mask_y + (y_min - (pointer_y - 10))
                        mask_x = mask_x + (x_min - (pointer_x - 10))
                        
                        valid_indices = (
                            (mask_y >= 0) & 
                            (mask_y < current_mask.shape[0]) & 
                            (mask_x >= 0) & 
                            (mask_x < current_mask.shape[1])
                        )
                        
                        mask_y = mask_y[valid_indices]
                        mask_x = mask_x[valid_indices]
                        
                        current_mask[mask_y, mask_x] = 1.0
                    
                    # Update observation
                    next_obs = np.zeros_like(obs)
                    next_obs[0:3] = image  # RGB image
                    next_obs[3] = current_mask  # Updated mask
                    
                    # Create pointer position channel
                    pointer_channel = np.zeros_like(current_mask)
                    
                    # Create a simpler pointer marker
                    y_min = max(0, pointer_y - 5)
                    y_max = min(pointer_channel.shape[0], pointer_y + 6)
                    x_min = max(0, pointer_x - 5)
                    x_max = min(pointer_channel.shape[1], pointer_x + 6)
                    
                    # Set a simple square marker for pointer
                    pointer_channel[y_min:y_max, x_min:x_max] = 1.0
                    next_obs[4] = pointer_channel
                    
                    # Create distance map and edge map
                    dist_map = np.zeros_like(current_mask)
                    edge_map = np.zeros_like(current_mask)
                    if current_mask.sum() > 0:
                        # Simple distance transform
                        from scipy.ndimage import distance_transform_edt
                        dist_map = distance_transform_edt(current_mask.squeeze() > 0.5)
                        if dist_map.max() > 0:
                            dist_map = dist_map / dist_map.max()
                        
                        # Simple edge detection
                        import cv2
                        binary_mask = (current_mask.squeeze() > 0.5).astype(np.uint8) * 255
                        if binary_mask.ndim == 2:  # Ensure mask is 2D for Canny
                            edge_map = cv2.Canny(binary_mask, 100, 200) / 255.0
                        else:
                            edge_map = np.zeros_like(binary_mask)
                    
                    # Ensure shapes match observation
                    dist_map = dist_map.reshape(current_mask.shape)
                    edge_map = edge_map.reshape(current_mask.shape)
                    
                    next_obs[5] = dist_map
                    next_obs[6] = edge_map
                    
                    next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(agent.device)
                    
                    # Update variables
                    obs = next_obs
                    obs_tensor = next_obs_tensor
                    step += 1
                
                # Calculate metrics
                dice = dice_coefficient(current_mask, gt_mask)
                iou = iou_score(current_mask, gt_mask)
                
                val_dice_scores.append(dice)
                val_iou_scores.append(iou)
            
            # Calculate average metrics
            avg_val_dice = np.mean(val_dice_scores)
            avg_val_iou = np.mean(val_iou_scores)
            
            print(f"Validation: Dice: {avg_val_dice:.4f}, IoU: {avg_val_iou:.4f}")
            
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
    plt.plot(episode_dice_scores)
    plt.title('Dice Scores')
    plt.xlabel('Episode')
    plt.ylabel('Dice')
    
    plt.subplot(1, 3, 3)
    plt.plot(episode_iou_scores)
    plt.title('IoU Scores')
    plt.xlabel('Episode')
    plt.ylabel('IoU')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'training_metrics.png'))
    plt.close()

if __name__ == '__main__':
    import cv2  # Import here for environment simulation
    args = parse_args()
    train(args) 