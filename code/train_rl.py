import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
from tqdm import tqdm

# Add code directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.data_utils import PolypDataset, get_data_loaders
from code.rl_environment import PolypSegmentationEnv, PolypFeatureExtractor
from code.rl_agent import PPOAgent

def train_agent(agent, train_dataset, device, num_episodes=1000, update_interval=128, save_interval=100):
    """
    Train PPO agent
    
    Args:
        agent (PPOAgent): PPO agent
        train_dataset (PolypDataset): Training dataset
        device (str): Device to use for computations
        num_episodes (int): Number of episodes
        update_interval (int): Update interval
        save_interval (int): Save interval
        
    Returns:
        dict: Training history
    """
    history = {
        'episode_rewards': [],
        'episode_dice': [],
        'episode_steps': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': []
    }
    
    # Create directory for saving models
    os.makedirs('models', exist_ok=True)
    
    # Training loop
    episode_count = 0
    step_count = 0
    
    pbar = tqdm(total=num_episodes)
    
    while episode_count < num_episodes:
        # Sample an image-mask pair from the dataset
        sample_idx = random.randint(0, len(train_dataset) - 1)
        sample = train_dataset[sample_idx]
        
        # Create environment
        env = PolypSegmentationEnv(
            image=sample['image'],
            gt_mask=sample['mask'],
            max_steps=100,
            step_size=10,
            device=device
        )
        
        # Reset environment
        state = env.reset()
        
        # Episode variables
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # Episode loop
        while not done:
            # Get action
            action, log_prob, value = agent.get_action(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Update episode variables
            episode_reward += reward
            episode_steps += 1
            step_count += 1
            
            # Store experience
            agent.remember(state, action, log_prob, reward, value, done)
            
            # Update state
            state = next_state
            
            # Update agent if enough steps
            if step_count % update_interval == 0:
                # Get next value (for advantage estimation)
                _, _, next_value = agent.get_action(state)
                
                # Update agent
                policy_loss, value_loss, entropy = agent.update(next_value)
                
                # Store losses
                history['policy_losses'].append(policy_loss)
                history['value_losses'].append(value_loss)
                history['entropies'].append(entropy)
                
                print(f'Episode {episode_count}, Step {step_count}: '
                      f'Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}')
        
        # Store episode results
        history['episode_rewards'].append(episode_reward)
        history['episode_dice'].append(info['dice'])
        history['episode_steps'].append(episode_steps)
        
        # Print episode results
        print(f'Episode {episode_count}: '
              f'Reward: {episode_reward:.4f}, Dice: {info["dice"]:.4f}, Steps: {episode_steps}')
        
        # Save model
        if episode_count % save_interval == 0:
            agent.save(f'models/ppo_agent_episode_{episode_count}.pth')
        
        # Increment episode count
        episode_count += 1
        pbar.update(1)
    
    # Save final model
    agent.save('models/ppo_agent_final.pth')
    
    pbar.close()
    
    return history

def evaluate_agent(agent, test_dataset, device, num_episodes=50):
    """
    Evaluate PPO agent
    
    Args:
        agent (PPOAgent): PPO agent
        test_dataset (PolypDataset): Test dataset
        device (str): Device to use for computations
        num_episodes (int): Number of episodes
        
    Returns:
        dict: Evaluation results
    """
    results = {
        'dice_scores': [],
        'iou_scores': [],
        'steps': []
    }
    
    # Create directory for saving results
    os.makedirs('results', exist_ok=True)
    
    # Sample random images from test set
    sample_indices = random.sample(range(len(test_dataset)), min(num_episodes, len(test_dataset)))
    
    for i, sample_idx in enumerate(sample_indices):
        # Get sample
        sample = test_dataset[sample_idx]
        
        # Create environment
        env = PolypSegmentationEnv(
            image=sample['image'],
            gt_mask=sample['mask'],
            max_steps=100,
            step_size=10,
            device=device
        )
        
        # Reset environment
        state = env.reset()
        
        # Store states for visualization
        image_states = []
        mask_states = []
        dice_states = []
        
        # Episode loop
        done = False
        
        while not done:
            # Get action (use eval mode)
            action, _, _ = agent.get_action(state, eval_mode=True)
            
            # Store current state
            image_states.append(state['image'].copy())
            mask_states.append(state['mask'].copy())
            dice_states.append(env._calculate_dice())
            
            # Take step in environment
            state, _, done, info = env.step(action)
        
        # Store final state
        image_states.append(state['image'].copy())
        mask_states.append(state['mask'].copy())
        dice_states.append(env._calculate_dice())
        
        # Store results
        results['dice_scores'].append(info['dice'])
        results['iou_scores'].append(info['iou'])
        results['steps'].append(info['steps'])
        
        # Visualize episode
        if i < 5:  # Only visualize first 5 episodes
            # Create directory for episode visualizations
            episode_dir = f'results/episode_{i}'
            os.makedirs(episode_dir, exist_ok=True)
            
            # Plot initial, middle, and final states
            steps = len(image_states)
            indices = [0, steps // 2, steps - 1]
            
            fig, axes = plt.subplots(len(indices), 3, figsize=(15, 5 * len(indices)))
            
            # 确保ground truth掩码是2D数组
            gt_mask = sample['mask'].cpu().numpy()
            if gt_mask.ndim == 3:  # 如果是3D的，例如(1, H, W)
                gt_mask = gt_mask.squeeze(0)
            
            for j, idx in enumerate(indices):
                # Original image
                axes[j, 0].imshow(image_states[idx])
                axes[j, 0].set_title(f'Step {idx}')
                axes[j, 0].axis('off')
                
                # Current mask
                axes[j, 1].imshow(mask_states[idx], cmap='gray')
                axes[j, 1].set_title(f'Mask (Dice: {dice_states[idx]:.4f})')
                axes[j, 1].axis('off')
                
                # Ground truth
                axes[j, 2].imshow(gt_mask, cmap='gray')
                axes[j, 2].set_title('Ground Truth')
                axes[j, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{episode_dir}/visualization.png')
            plt.close()
            
            # Plot dice progression
            plt.figure(figsize=(10, 5))
            plt.plot(dice_states)
            plt.xlabel('Step')
            plt.ylabel('Dice Coefficient')
            plt.title('Dice Coefficient Progression')
            plt.grid(True)
            plt.savefig(f'{episode_dir}/dice_progression.png')
            plt.close()
    
    # Calculate average results
    avg_dice = np.mean(results['dice_scores'])
    avg_iou = np.mean(results['iou_scores'])
    avg_steps = np.mean(results['steps'])
    
    print(f'Evaluation Results:')
    print(f'Average Dice: {avg_dice:.4f}')
    print(f'Average IoU: {avg_iou:.4f}')
    print(f'Average Steps: {avg_steps:.2f}')
    
    return results

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Set data directory
    data_dir = '../data/raw'
    
    # Get data loaders
    data_loaders = get_data_loaders(data_dir, batch_size=4, num_workers=2, seed=42)
    
    # Create feature extractor
    feature_extractor = PolypFeatureExtractor(device=device)
    
    # Set input dimension (extracted features + pointer location)
    input_dim = 4 * 16 * 16 + 2  # Features from a 4-channel 16x16 feature map + 2 for pointer location
    
    # Create PPO agent
    agent = PPOAgent(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=7,  # 7 actions
        feature_extractor=feature_extractor,
        device=device,
        lr=3e-4,
        gamma=0.99,
        clip_ratio=0.2,
        entropy_coef=0.01
    )
    
    # Train agent
    history = train_agent(
        agent=agent,
        train_dataset=data_loaders['train'].dataset,
        device=device,
        num_episodes=1000,
        update_interval=128,
        save_interval=100
    )
    
    # Plot training history
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(2, 2, 2)
    plt.plot(history['episode_dice'])
    plt.title('Episode Dice Coefficient')
    plt.xlabel('Episode')
    plt.ylabel('Dice')
    
    plt.subplot(2, 2, 3)
    plt.plot(history['episode_steps'])
    plt.title('Episode Steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.subplot(2, 2, 4)
    plt.plot(history['policy_losses'], label='Policy Loss')
    plt.plot(history['value_losses'], label='Value Loss')
    plt.plot(history['entropies'], label='Entropy')
    plt.title('Losses and Entropy')
    plt.xlabel('Update')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../results/rl_training_history.png')
    plt.close()
    
    # Evaluate agent
    results = evaluate_agent(
        agent=agent,
        test_dataset=data_loaders['test'].dataset,
        device=device,
        num_episodes=50
    )
    
    # Compare with U-Net baseline (assuming we have U-Net results)
    try:
        # Load U-Net results
        unet_dice = 0.8  # Example value, replace with actual U-Net results
        unet_iou = 0.7   # Example value, replace with actual U-Net results
        
        # Create comparison plot
        plt.figure(figsize=(10, 6))
        
        x = ['Dice', 'IoU']
        unet_values = [unet_dice, unet_iou]
        rl_values = [np.mean(results['dice_scores']), np.mean(results['iou_scores'])]
        
        x_pos = np.arange(len(x))
        width = 0.35
        
        plt.bar(x_pos - width/2, unet_values, width, label='U-Net')
        plt.bar(x_pos + width/2, rl_values, width, label='RL')
        
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('U-Net vs RL Comparison')
        plt.xticks(x_pos, x)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('../results/unet_vs_rl_comparison.png')
        plt.close()
    except:
        print('Could not create comparison plot. U-Net results not available.')
    
if __name__ == '__main__':
    main() 