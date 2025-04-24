import os
import sys
import argparse
import torch
import numpy as np
import random
from pathlib import Path
import cv2
import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import pickle

# Add the code directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from code.data_utils import PolypDataset, get_data_loaders, visualize_sample
from code.unet_model import UNet, train_unet
from code.rl_environment import PolypSegmentationEnv, PolypFeatureExtractor
from code.rl_agent import PPOAgent

def setup_environment():
    """Set up the environment (directories, random seeds, etc.)"""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return device

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Reinforcement Learning for Interactive Polyp Segmentation')
    
    parser.add_argument('--mode', type=str, default='all', choices=['train_unet', 'train_rl', 'train_simple_rl', 'evaluate', 'all', 'compare_unet_simple_rl', 'train_and_evaluate'],
                        help='Mode of operation: train_unet (只训练U-Net), train_rl (训练PPO代理), train_simple_rl (训练InteractiveRL代理并保存训练历史), evaluate (评估), all (全部), compare_unet_simple_rl (比较UNet和SimpleRL), train_and_evaluate (训练并评估)')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Directory containing the dataset')
    parser.add_argument('--unet_epochs', type=int, default=30,
                        help='Number of epochs for training U-Net')
    parser.add_argument('--rl_episodes', type=int, default=1000,
                        help='Number of episodes for training RL agent')
    parser.add_argument('--simple_rl_episodes', type=int, default=50,
                        help='Number of episodes for training SimpleRL agent (50-100建议值，用于生成训练曲线)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=20,
                        help='Maximum steps per episode for RL')
    parser.add_argument('--eval_samples', type=int, default=50,
                        help='Number of samples to evaluate on')
    parser.add_argument('--save_visualizations', action='store_true',
                        help='Whether to save visualization images during evaluation')
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Interval for validation evaluation during InteractiveRL training')
    parser.add_argument('--num_eval_episodes', type=int, default=5,
                        help='Number of evaluation episodes during InteractiveRL training validation')
    
    return parser.parse_args()

def train_unet_model(args, device, data_loaders):
    """Train and evaluate the U-Net model"""
    print('Training U-Net model...')
    
    # Create model
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    
    # Train model
    history = train_unet(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        device=device,
        epochs=args.unet_epochs,
        lr=args.lr
    )
    
    # Save model
    torch.save(model.state_dict(), 'models/unet_model.pth')
    
    # Return the trained model
    return model, history

def train_rl_agent(args, device, data_loaders):
    """Train and evaluate the RL agent"""
    from code.train_rl import train_agent, evaluate_agent
    
    print('Training RL agent...')
    
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
        lr=args.lr,
        gamma=0.99,
        clip_ratio=0.2,
        entropy_coef=0.01
    )
    
    # Train agent
    history = train_agent(
        agent=agent,
        train_dataset=data_loaders['train'].dataset,
        device=device,
        num_episodes=args.rl_episodes,
        update_interval=128,
        save_interval=100
    )
    
    # Return the trained agent
    return agent, history

def train_simple_rl_agent(args, device, data_loaders):
    """Train the SimpleRL agent from simple_rl.py"""
    from code.simple_rl import SimpleRLAgent, train_agent
    
    print('Training Simple RL agent...')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建代理
    agent = SimpleRLAgent(
        n_actions=7,  # 0=Up, 1=Down, 2=Left, 3=Right, 4=Expand, 5=Shrink, 6=Done
        lr=args.lr,
        gamma=0.99,
        device=device
    )
    
    # 直接使用传入的args参数
    train_agent(args)
    
    # 返回代理
    return agent

def evaluate_unet(model, test_dataset, device, save_dir=None, num_samples=50, save_visualizations=False):
    """Evaluate the U-Net model on the test dataset and save results"""
    from code.unet_model import dice_coefficient, iou_score
    import torch
    import matplotlib.pyplot as plt
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if save_visualizations:
            os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)
    
    model.eval()
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            # Get sample
            sample = test_dataset[i]
            image = sample['image'].unsqueeze(0).to(device)
            gt_mask = sample['mask'].unsqueeze(0).to(device)
            
            # Predict mask
            output = model(image)
            pred_mask = (output > 0.5).float()  # Use 0.5 threshold for sigmoid output
            
            # Calculate metrics
            dice = dice_coefficient(pred_mask, gt_mask).item()
            dice_scores.append(dice)
            
            iou = iou_score(pred_mask, gt_mask).item()
            iou_scores.append(iou)
            
            # Save visualizations if requested
            if save_visualizations and save_dir and i < 5:  # Save only first 5 visualizations
                # Convert tensors to numpy for visualization
                image_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
                gt_mask_np = gt_mask.squeeze().cpu().numpy()
                pred_mask_np = pred_mask.squeeze().cpu().numpy()
                
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                plt.imshow(image_np)
                plt.title('Original Image')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(gt_mask_np, cmap='gray')
                plt.title('Ground Truth')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(pred_mask_np, cmap='gray')
                plt.title(f'Predicted Mask\nDice: {dice:.4f}, IoU: {iou:.4f}')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'visualizations', f'unet_sample_{i+1}.png'))
                plt.close()
    
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    
    results = {
        'dice_scores': dice_scores,
        'iou_scores': iou_scores,
        'mean_dice': mean_dice,
        'mean_iou': mean_iou
    }
    
    # Save results to a text file if save_dir is provided
    if save_dir:
        with open(os.path.join(save_dir, 'unet_evaluation_results.txt'), 'w') as f:
            f.write(f'U-Net Evaluation Results\n')
            f.write(f'=======================\n\n')
            f.write(f'Mean Dice Score: {mean_dice:.4f}\n')
            f.write(f'Mean IoU Score: {mean_iou:.4f}\n\n')
            f.write(f'Individual Sample Scores:\n')
            for i, (dice, iou) in enumerate(zip(dice_scores, iou_scores)):
                f.write(f'Sample {i+1}: Dice = {dice:.4f}, IoU = {iou:.4f}\n')
    
    return results

def evaluate_simple_rl(agent, test_dataset, device, max_steps=20, save_dir=None, num_samples=50, save_visualizations=False):
    """Evaluate the SimpleRL agent on the test dataset and save results"""
    from code.simple_rl import predict_mask, visualize_results
    from code.utils import dice_coefficient, iou_score
    import matplotlib.pyplot as plt
    import os
    import torch.nn.functional as F
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if save_visualizations:
            os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)
    
    dice_scores = []
    iou_scores = []
    
    for i in range(min(num_samples, len(test_dataset))):
        try:
            # Get sample
            sample = test_dataset[i]
            image = sample['image'].numpy()
            gt_mask = sample['mask'].numpy()
            
            # Check and ensure the image has the expected shape
            expected_shape = (3, 288, 384)  # The shape expected by the SimplePolicyNetwork
            
            # Print the original shape for debugging
            print(f"Sample {i+1} image shape: {image.shape}")
            
            if image.shape != expected_shape:
                print(f"Resizing image from {image.shape} to {expected_shape}")
                # Transpose to (H, W, C) for resizing if needed
                if image.shape[0] == 3:
                    # Already in (C, H, W) format
                    image_for_resize = np.transpose(image, (1, 2, 0))
                    resized_image = cv2.resize(image_for_resize, (expected_shape[2], expected_shape[1]))
                    image = np.transpose(resized_image, (2, 0, 1))
                else:
                    # Assumed to be in (H, W, C) format
                    resized_image = cv2.resize(image, (expected_shape[2], expected_shape[1]))
                    if resized_image.ndim == 2:
                        # If grayscale, add channel dimension
                        resized_image = np.expand_dims(resized_image, axis=0)
                    else:
                        # If color, transpose to (C, H, W)
                        image = np.transpose(resized_image, (2, 0, 1))
                
                # Also resize the mask
                if gt_mask.shape != (1, expected_shape[1], expected_shape[2]):
                    mask_for_resize = gt_mask.squeeze()
                    resized_mask = cv2.resize(mask_for_resize, (expected_shape[2], expected_shape[1]))
                    gt_mask = np.expand_dims(resized_mask, axis=0)
            
            # Predict mask
            pred_mask = predict_mask(agent, image, gt_mask, max_steps=max_steps)
            
            # Calculate metrics
            dice = dice_coefficient(torch.tensor(pred_mask), torch.tensor(gt_mask))
            dice_scores.append(dice.item())
            
            iou = iou_score(torch.tensor(pred_mask), torch.tensor(gt_mask))
            iou_scores.append(iou.item())
            
            # Visualize samples if requested
            if save_visualizations and save_dir and i < 5:  # Only visualize the first 5 samples
                vis_img = visualize_results(image, gt_mask, pred_mask)
                plt.figure(figsize=(8, 8))
                plt.imshow(vis_img)
                plt.title(f'SimpleRL Segmentation\nDice: {dice.item():.4f}, IoU: {iou.item():.4f}')
                plt.axis('off')
                plt.savefig(os.path.join(save_dir, 'visualizations', f'simple_rl_sample_{i+1}.png'), bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"Error processing sample {i+1}: {str(e)}")
            continue
    
    if not dice_scores:
        print("No valid predictions were made. Check if the model is compatible with the input data.")
        return {
            'dice_scores': [],
            'iou_scores': [],
            'mean_dice': 0.0,
            'mean_iou': 0.0
        }
    
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    
    results = {
        'dice_scores': dice_scores,
        'iou_scores': iou_scores,
        'mean_dice': mean_dice,
        'mean_iou': mean_iou
    }
    
    # Save results to a text file if save_dir is provided
    if save_dir:
        with open(os.path.join(save_dir, 'simple_rl_evaluation_results.txt'), 'w') as f:
            f.write(f'SimpleRL Evaluation Results\n')
            f.write(f'==========================\n\n')
            f.write(f'Mean Dice Score: {mean_dice:.4f}\n')
            f.write(f'Mean IoU Score: {mean_iou:.4f}\n\n')
            f.write(f'Individual Sample Scores:\n')
            for i, (dice, iou) in enumerate(zip(dice_scores, iou_scores)):
                f.write(f'Sample {i+1}: Dice = {dice:.4f}, IoU = {iou:.4f}\n')
    
    return results

def compare_unet_simple_rl(args, device, data_loaders):
    """Compare U-Net and SimpleRL on the same dataset"""
    print('Comparing U-Net and SimpleRL...')
    
    results = {}
    
    # Train U-Net for specified epochs
    unet_model, unet_history = train_unet_model(args, device, data_loaders)
    
    # Evaluate U-Net
    from code.unet_model import dice_coefficient, iou_score
    
    unet_dice = 0
    unet_iou = 0
    
    with torch.no_grad():
        for batch in data_loaders['test']:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = unet_model(images)
            pred_masks = (outputs > 0.5).float()
            
            unet_dice += dice_coefficient(pred_masks, masks).item()
            unet_iou += iou_score(pred_masks, masks).item()
    
    unet_dice /= len(data_loaders['test'])
    unet_iou /= len(data_loaders['test'])
    
    results['unet'] = {
        'dice': unet_dice,
        'iou': unet_iou
    }
    
    print(f'U-Net Results:')
    print(f'  - Dice: {unet_dice:.4f}')
    print(f'  - IoU: {unet_iou:.4f}')
    
    # Train SimpleRL
    simple_rl_agent = train_simple_rl_agent(args, device, data_loaders)
    
    # Load the best model
    from code.simple_rl import SimpleRLAgent
    best_agent = SimpleRLAgent(
        n_actions=7,
        lr=args.lr,
        gamma=0.99,
        device=device
    )
    
    try:
        best_agent.load('results/simple_rl/best_model.pth')
        
        # Evaluate SimpleRL
        simple_rl_results = evaluate_simple_rl(
            agent=best_agent,
            test_dataset=data_loaders['test'].dataset,
            device=device,
            max_steps=args.max_steps
        )
        
        results['simple_rl'] = {
            'dice': simple_rl_results['mean_dice'],
            'iou': simple_rl_results['mean_iou']
        }
        
        print(f'SimpleRL Results:')
        print(f'  - Dice: {results["simple_rl"]["dice"]:.4f}')
        print(f'  - IoU: {results["simple_rl"]["iou"]:.4f}')
        
        # Compare results
        print('\nModel Comparison:')
        print(f'                U-Net      SimpleRL')
        print(f'Dice:           {results["unet"]["dice"]:.4f}      {results["simple_rl"]["dice"]:.4f}')
        print(f'IoU:            {results["unet"]["iou"]:.4f}      {results["simple_rl"]["iou"]:.4f}')
        
    except:
        print('Could not load best SimpleRL model. Skipping comparison.')
    
    return results

def evaluate_models(args, device, data_loaders, unet_model=None, rl_agent=None):
    """Evaluate and compare the U-Net model and RL agent"""
    from code.train_rl import evaluate_agent
    
    print('Evaluating models...')
    
    results = {}
    
    # Load models if not provided
    if unet_model is None:
        unet_model = UNet(n_channels=3, n_classes=1, bilinear=True)
        try:
            unet_model.load_state_dict(torch.load('models/unet_model.pth', map_location=device))
            unet_model = unet_model.to(device)
            unet_model.eval()
        except:
            print('Could not load U-Net model. Skipping U-Net evaluation.')
            unet_model = None
    
    if rl_agent is None:
        feature_extractor = PolypFeatureExtractor(device=device)
        input_dim = 4 * 16 * 16 + 2
        
        rl_agent = PPOAgent(
            input_dim=input_dim,
            hidden_dim=128,
            output_dim=7,
            feature_extractor=feature_extractor,
            device=device
        )
        
        try:
            rl_agent.load('models/ppo_agent_final.pth')
        except:
            print('Could not load RL agent. Skipping RL evaluation.')
            rl_agent = None
    
    # Evaluate U-Net model
    if unet_model is not None:
        from code.unet_model import dice_coefficient, iou_score
        
        unet_dice = 0
        unet_iou = 0
        
        with torch.no_grad():
            for batch in data_loaders['test']:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = unet_model(images)
                pred_masks = (outputs > 0.5).float()
                
                unet_dice += dice_coefficient(pred_masks, masks).item()
                unet_iou += iou_score(pred_masks, masks).item()
        
        unet_dice /= len(data_loaders['test'])
        unet_iou /= len(data_loaders['test'])
        
        results['unet'] = {
            'dice': unet_dice,
            'iou': unet_iou
        }
        
        print(f'U-Net Results:')
        print(f'  - Dice: {unet_dice:.4f}')
        print(f'  - IoU: {unet_iou:.4f}')
    
    # Evaluate RL agent
    if rl_agent is not None:
        rl_results = evaluate_agent(
            agent=rl_agent,
            test_dataset=data_loaders['test'].dataset,
            device=device,
            num_episodes=50
        )
        
        results['rl'] = {
            'dice': np.mean(rl_results['dice_scores']),
            'iou': np.mean(rl_results['iou_scores']),
            'steps': np.mean(rl_results['steps'])
        }
        
        print(f'RL Agent Results:')
        print(f'  - Dice: {results["rl"]["dice"]:.4f}')
        print(f'  - IoU: {results["rl"]["iou"]:.4f}')
        print(f'  - Average Steps: {results["rl"]["steps"]:.2f}')
    
    # Compare models
    if 'unet' in results and 'rl' in results:
        print('\nModel Comparison:')
        print(f'                U-Net      RL Agent')
        print(f'Dice:           {results["unet"]["dice"]:.4f}      {results["rl"]["dice"]:.4f}')
        print(f'IoU:            {results["unet"]["iou"]:.4f}      {results["rl"]["iou"]:.4f}')
        
        # Additional RL metrics
        print(f'\nAdditional RL Metrics:')
        print(f'Average Steps: {results["rl"]["steps"]:.2f}')
    
    return results

def train_and_evaluate(args, device, data_loaders):
    """Train both U-Net and SimpleRL models, save them, and then evaluate and compare them"""
    # 导入必要的函数
    from code.unet_model import dice_coefficient, iou_score, DiceLoss
    from code.simple_rl import train_agent
    import numpy as np
    import pickle
    
    print('Training and evaluating models...')
    
    # 创建结果目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'results/comparison_{timestamp}'
    models_dir = f'models/checkpoints_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(models_dir, 'unet'), exist_ok=True)
    os.makedirs(os.path.join(models_dir, 'simple_rl'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'unet_data'), exist_ok=True)
    
    # 保存训练配置
    with open(os.path.join(results_dir, 'training_config.txt'), 'w') as f:
        f.write(f'Training Configuration\n')
        f.write(f'======================\n\n')
        f.write(f'U-Net Epochs: {args.unet_epochs}\n')
        f.write(f'SimpleRL Episodes: {args.simple_rl_episodes}\n')
        f.write(f'Max Steps per Episode: {args.max_steps}\n')
        f.write(f'Batch Size: {args.batch_size}\n')
        f.write(f'Learning Rate: {args.lr}\n')
        f.write(f'Device: {device}\n')
        f.write(f'Timestamp: {timestamp}\n')
    
    # 1. 训练U-Net模型并保存检查点
    print('\n1. Training U-Net model...')
    
    # 创建U-Net模型
    unet_model = UNet(n_channels=3, n_classes=1, bilinear=True)
    unet_model = unet_model.to(device)
    
    # 创建U-Net优化器
    optimizer = torch.optim.AdamW(unet_model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 使用组合损失函数：BCE + Dice Loss
    bce_criterion = nn.BCELoss()
    dice_criterion = DiceLoss()
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=True
    )
    
    # 用于早停和保存最佳模型
    best_val_dice = 0.0
    best_epoch = 0
    patience = 15  # 早停耐心值
    patience_counter = 0
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'lr': [],
        'epochs': [],
        'time_per_epoch': [],
        'batch_losses': [],  # 记录每个批次的损失
        'per_sample_dice': [],  # 每个样本的Dice分数
        'per_sample_iou': []    # 每个样本的IoU分数
    }
    
    for epoch in range(args.unet_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        unet_model.train()
        train_loss = 0
        batch_losses = []
        
        for batch_idx, batch in enumerate(data_loaders['train']):
            # 获取图像和掩码
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # 正向传播
            outputs = unet_model(images)
            
            # 计算损失 - 组合BCE和Dice Loss
            bce_loss = bce_criterion(outputs, masks)
            dice_loss = dice_criterion(outputs, masks)
            loss = bce_loss + dice_loss
            
            # 记录每个批次的损失
            batch_losses.append(loss.item())
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet_model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            
            train_loss += loss.item()
            
            # 每10个批次打印一次进度
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.unet_epochs} | Batch {batch_idx}/{len(data_loaders['train'])} | Loss: {loss.item():.4f}")
        
        train_loss /= len(data_loaders['train'])
        history['train_loss'].append(train_loss)
        history['batch_losses'].append(batch_losses)
        
        # 验证阶段
        unet_model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        epoch_per_sample_dice = []
        epoch_per_sample_iou = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loaders['val']):
                # 获取图像和掩码
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # 正向传播
                outputs = unet_model(images)
                
                # 计算损失
                bce_loss = bce_criterion(outputs, masks)
                dice_loss = dice_criterion(outputs, masks)
                loss = bce_loss + dice_loss
                val_loss += loss.item()
                
                # 生成二值掩码预测，注意阈值是0.5
                pred_masks = (outputs > 0.5).float()
                
                # 逐样本计算指标
                for i in range(pred_masks.size(0)):
                    sample_dice = dice_coefficient(pred_masks[i:i+1], masks[i:i+1]).item()
                    sample_iou = iou_score(pred_masks[i:i+1], masks[i:i+1]).item()
                    
                    epoch_per_sample_dice.append(sample_dice)
                    epoch_per_sample_iou.append(sample_iou)
                
                # 计算批次级指标
                batch_dice = dice_coefficient(pred_masks, masks).item()
                batch_iou = iou_score(pred_masks, masks).item()
                
                val_dice += batch_dice
                val_iou += batch_iou
        
        val_loss /= len(data_loaders['val'])
        val_dice /= len(data_loaders['val'])
        val_iou /= len(data_loaders['val'])
        
        # 更新学习率调度器
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # 记录每个epoch的训练时间
        epoch_time = time.time() - epoch_start_time
        
        # 记录验证指标
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        history['lr'].append(current_lr)
        history['epochs'].append(epoch + 1)
        history['time_per_epoch'].append(epoch_time)
        history['per_sample_dice'].append(epoch_per_sample_dice)
        history['per_sample_iou'].append(epoch_per_sample_iou)
        
        print(f"Epoch {epoch+1}/{args.unet_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, Time: {epoch_time:.2f}s")
        
        # 保存每个epoch的训练数据
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'val_iou': val_iou,
            'lr': current_lr,
            'time': epoch_time,
            'batch_losses': batch_losses,
            'per_sample_dice': epoch_per_sample_dice,
            'per_sample_iou': epoch_per_sample_iou
        }
        
        # 保存每个epoch的详细数据
        with open(os.path.join(results_dir, 'unet_data', f'epoch_{epoch+1}_data.pkl'), 'wb') as f:
            pickle.dump(epoch_data, f)
        
        # 可视化当前epoch的验证结果
        if args.save_visualizations:
            # 随机选择几个验证样本进行可视化
            with torch.no_grad():
                sample_indices = np.random.randint(0, len(data_loaders['val'].dataset), 3)
                
                for i, idx in enumerate(sample_indices):
                    sample = data_loaders['val'].dataset[idx]
                    image = sample['image'].unsqueeze(0).to(device)
                    gt_mask = sample['mask'].unsqueeze(0).to(device)
                    
                    output = unet_model(image)
                    pred_mask = (output > 0.5).float()
                    
                    # 计算指标
                    sample_dice = dice_coefficient(pred_mask, gt_mask).item()
                    sample_iou = iou_score(pred_mask, gt_mask).item()
                    
                    # 转换为numpy
                    image_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
                    gt_mask_np = gt_mask.squeeze().cpu().numpy()
                    pred_mask_np = pred_mask.squeeze().cpu().numpy()
                    
                    plt.figure(figsize=(12, 4))
                    
                    plt.subplot(1, 3, 1)
                    plt.imshow(image_np)
                    plt.title('Original Image')
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 2)
                    plt.imshow(gt_mask_np, cmap='gray')
                    plt.title('Ground Truth')
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 3)
                    plt.imshow(pred_mask_np, cmap='gray')
                    plt.title(f'Predicted Mask\nDice: {sample_dice:.4f}, IoU: {sample_iou:.4f}')
                    plt.axis('off')
                    
                    plt.tight_layout()
                    os.makedirs(os.path.join(results_dir, 'unet_data', f'epoch_{epoch+1}_viz'), exist_ok=True)
                    plt.savefig(os.path.join(results_dir, 'unet_data', f'epoch_{epoch+1}_viz', f'sample_{i+1}.png'))
                    plt.close()
        
        # 保存最佳模型，使用try-except块防止磁盘空间不足
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_path = os.path.join(models_dir, 'unet', 'unet_best_model.pth')
            
            # 使用try-except块保存最佳模型
            try:
                # 保存最佳模型
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': unet_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': val_loss,
                    'dice': val_dice,
                    'iou': val_iou
                }, best_model_path)
                
                print(f"保存新的最佳U-Net模型，验证Dice分数: {best_val_dice:.4f}")
            except Exception as e:
                print(f"保存最佳模型时出错: {str(e)}，跳过保存继续训练")
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= patience:
            print(f"早停在第{epoch+1}轮。{patience}轮内没有改进。")
            break
    
    # 保存最终模型，使用try-except块
    try:
        final_model_path = os.path.join(models_dir, 'unet', 'unet_final_model.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': unet_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_loss,
            'dice': val_dice,
            'iou': val_iou
        }, final_model_path)
    except Exception as e:
        print(f"保存最终模型时出错: {str(e)}，使用最佳模型继续")
    
    # 保存完整的训练历史记录
    with open(os.path.join(results_dir, 'unet_training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # 生成更多的训练可视化图表
    
    # 1. 损失与准确率曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['epochs'], history['train_loss'], label='Train Loss')
    plt.plot(history['epochs'], history['val_loss'], label='Validation Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch}')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['epochs'], history['val_dice'], label='Dice')
    plt.plot(history['epochs'], history['val_iou'], label='IoU')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch}')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['epochs'], history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'unet_training_curves.png'))
    plt.close()
    
    # 2. 每个epoch的训练时间条形图
    plt.figure(figsize=(10, 5))
    plt.bar(history['epochs'], history['time_per_epoch'])
    plt.title('Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'unet_training_time.png'))
    plt.close()
    
    # 3. 验证集上的分布图
    if len(history['per_sample_dice']) > 0:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        for epoch, dices in enumerate(history['per_sample_dice']):
            if epoch in [0, len(history['per_sample_dice'])//2, len(history['per_sample_dice'])-1]:
                plt.hist(dices, bins=20, alpha=0.7, label=f'Epoch {epoch+1}')
        plt.title('Dice Score Distribution (Validation)')
        plt.xlabel('Dice Score')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(1, 2, 2)
        for epoch, ious in enumerate(history['per_sample_iou']):
            if epoch in [0, len(history['per_sample_iou'])//2, len(history['per_sample_iou'])-1]:
                plt.hist(ious, bins=20, alpha=0.7, label=f'Epoch {epoch+1}')
        plt.title('IoU Score Distribution (Validation)')
        plt.xlabel('IoU Score')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'unet_score_distributions.png'))
        plt.close()
    
    # 加载最佳模型用于评估
    print(f"从第{best_epoch}轮加载最佳U-Net模型进行评估...")
    checkpoint = torch.load(best_model_path)
    unet_model.load_state_dict(checkpoint['model_state_dict'])
    unet_model.eval()
    print(f"加载了最佳U-Net模型，验证Dice: {checkpoint['dice']:.4f}, IoU: {checkpoint['iou']:.4f}")
    
    # 2. 训练SimpleRL模型
    print('\n2. Training SimpleRL model...')
    # 设置SimpleRL训练参数
    simple_rl_args = argparse.Namespace(
        data_dir=args.data_dir,
        output_dir=os.path.join(models_dir, 'simple_rl'),
        lr=args.lr,
        gamma=0.99,
        num_episodes=args.simple_rl_episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        update_interval=5,
        log_interval=5,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_episodes,
        save_interval=10,
        device=device,
        seed=42
    )
    
    # 直接调用simple_rl.py中的train_agent函数
    # 记录训练过程中的最佳性能
    train_start_time = time.time()
    simple_rl_training_results = train_agent(simple_rl_args)
    train_time = time.time() - train_start_time
    
    # 提取最佳验证性能（如果train_agent返回此信息）
    if isinstance(simple_rl_training_results, dict) and 'best_val_dice' in simple_rl_training_results:
        simple_rl_best_val_dice = simple_rl_training_results['best_val_dice']
        simple_rl_best_val_iou = simple_rl_training_results.get('best_val_iou', 0)
        simple_rl_best_episode = simple_rl_training_results.get('best_episode', 0)
        print(f'SimpleRL训练完成：最佳验证Dice={simple_rl_best_val_dice:.4f}, IoU={simple_rl_best_val_iou:.4f} at Episode {simple_rl_best_episode}')
    else:
        print('SimpleRL模型训练完成，但未返回最佳验证性能信息')
    
    print(f'SimpleRL模型训练用时：{train_time:.2f}秒')
    
    # 3. 加载模型进行评估
    print('\n3. Loading models for evaluation...')
    
    # U-Net模型已加载为最佳模型
    
    # 加载SimpleRL最佳模型
    simple_rl_agent = None
    from code.simple_rl import SimpleRLAgent
    try:
        print(f"正在加载SimpleRL最佳模型...")
        best_rl_path = os.path.join(models_dir, 'simple_rl', 'best_model.pth')
        
        # 检查最佳模型文件是否存在
        if os.path.isfile(best_rl_path):
            simple_rl_agent = SimpleRLAgent(
                n_actions=7,
                lr=args.lr,
                gamma=0.99,
                device=device
            )
            simple_rl_agent.load(best_rl_path)
            print('成功加载SimpleRL最佳模型')
            
            # 保存模型信息
            if isinstance(simple_rl_training_results, dict) and 'best_val_dice' in simple_rl_training_results:
                with open(os.path.join(results_dir, 'simple_rl_best_model_info.txt'), 'w') as f:
                    f.write(f'SimpleRL最佳模型信息\n')
                    f.write(f'===================\n\n')
                    f.write(f'Episode: {simple_rl_best_episode}\n')
                    f.write(f'验证Dice: {simple_rl_best_val_dice:.4f}\n')
                    f.write(f'验证IoU: {simple_rl_best_val_iou:.4f}\n')
                    f.write(f'模型路径: {best_rl_path}\n')
        else:
            print(f"警告：最佳模型文件不存在 ({best_rl_path})，尝试加载最终模型...")
            final_rl_path = os.path.join(models_dir, 'simple_rl', 'final_model.pth')
            if os.path.isfile(final_rl_path):
                simple_rl_agent = SimpleRLAgent(
                    n_actions=7,
                    lr=args.lr,
                    gamma=0.99,
                    device=device
                )
                simple_rl_agent.load(final_rl_path)
                print('成功加载SimpleRL最终模型')
            else:
                print(f"错误：最终模型文件也不存在 ({final_rl_path})")
                raise FileNotFoundError("找不到SimpleRL模型文件")
    except Exception as e:
        print(f'加载SimpleRL模型时出错: {str(e)}')
        print('SimpleRL模型无法加载，将跳过评估')
    
    # 4. 评估模型
    print('\n4. Evaluating models...')
    evaluation_results = {}
    
    # 评估U-Net
    print('Evaluating U-Net...')
    unet_results = evaluate_unet(
        model=unet_model,
        test_dataset=data_loaders['test'].dataset,
        device=device,
        save_dir=results_dir,
        num_samples=args.eval_samples,
        save_visualizations=args.save_visualizations
    )
    evaluation_results['unet'] = unet_results
    
    # 评估SimpleRL（如果可用）
    if simple_rl_agent:
        print('Evaluating SimpleRL...')
        # 在评估时使用与训练相同的max_steps参数
        simple_rl_results = evaluate_simple_rl(
            agent=simple_rl_agent,
            test_dataset=data_loaders['test'].dataset,
            device=device,
            max_steps=args.max_steps,
            save_dir=results_dir,
            num_samples=args.eval_samples,
            save_visualizations=args.save_visualizations
        )
        evaluation_results['simple_rl'] = simple_rl_results
        
        # 将训练验证集上的最佳性能也添加到评估结果中
        if isinstance(simple_rl_training_results, dict) and 'best_val_dice' in simple_rl_training_results:
            simple_rl_results['best_val_dice'] = simple_rl_training_results['best_val_dice']
            simple_rl_results['best_val_iou'] = simple_rl_training_results.get('best_val_iou', 0)
            simple_rl_results['best_episode'] = simple_rl_training_results.get('best_episode', 0)
    
    # 5. 对比结果
    if 'unet' in evaluation_results and 'simple_rl' in evaluation_results:
        print('\n5. Comparison of results:')
        print(f'                U-Net      SimpleRL')
        print(f'Dice:           {evaluation_results["unet"]["mean_dice"]:.4f}      {evaluation_results["simple_rl"]["mean_dice"]:.4f}')
        print(f'IoU:            {evaluation_results["unet"]["mean_iou"]:.4f}      {evaluation_results["simple_rl"]["mean_iou"]:.4f}')
        
        # 如果有训练验证集上的最佳性能，也打印出来
        if 'best_val_dice' in evaluation_results['simple_rl']:
            print(f'\nSimpleRL在验证集上的最佳性能（第{evaluation_results["simple_rl"]["best_episode"]}轮）:')
            print(f'验证Dice: {evaluation_results["simple_rl"]["best_val_dice"]:.4f}')
            print(f'验证IoU: {evaluation_results["simple_rl"]["best_val_iou"]:.4f}')
        
        # 保存对比结果到文件
        with open(os.path.join(results_dir, 'model_comparison.txt'), 'w') as f:
            f.write(f'Model Comparison\n')
            f.write(f'===============\n\n')
            f.write(f'                U-Net      SimpleRL\n')
            f.write(f'Dice:           {evaluation_results["unet"]["mean_dice"]:.4f}      {evaluation_results["simple_rl"]["mean_dice"]:.4f}\n')
            f.write(f'IoU:            {evaluation_results["unet"]["mean_iou"]:.4f}      {evaluation_results["simple_rl"]["mean_iou"]:.4f}\n')
            
            # 添加训练验证集上的最佳性能
            if 'best_val_dice' in evaluation_results['simple_rl']:
                f.write(f'\nSimpleRL在验证集上的最佳性能（第{evaluation_results["simple_rl"]["best_episode"]}轮）:\n')
                f.write(f'验证Dice: {evaluation_results["simple_rl"]["best_val_dice"]:.4f}\n')
                f.write(f'验证IoU: {evaluation_results["simple_rl"]["best_val_iou"]:.4f}\n')
            
        # 生成对比图
        plt.figure(figsize=(8, 6))
        models = ['U-Net', 'SimpleRL']
        dice_scores = [evaluation_results['unet']['mean_dice'], evaluation_results['simple_rl']['mean_dice']]
        iou_scores = [evaluation_results['unet']['mean_iou'], evaluation_results['simple_rl']['mean_iou']]
        
        # 添加SimpleRL验证集上的最佳性能
        if 'best_val_dice' in evaluation_results['simple_rl']:
            models.append('SimpleRL\n(验证集最佳)')
            dice_scores.append(evaluation_results['simple_rl']['best_val_dice'])
            iou_scores.append(evaluation_results['simple_rl']['best_val_iou'])
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, dice_scores, width, label='Dice')
        rects2 = ax.bar(x + width/2, iou_scores, width, label='IoU')
        
        ax.set_ylabel('Score')
        ax.set_title('Segmentation Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        
        # 添加数值标签
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        plt.savefig(os.path.join(results_dir, 'performance_comparison.png'))
        plt.close()
    
    return evaluation_results, models_dir, results_dir

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set up environment
    device = setup_environment()
    print(f'Using device: {device}')
    
    # Get data loaders
    data_loaders = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2,
        seed=42
    )
    
    # Based on the mode, run the appropriate functions
    unet_model = None
    rl_agent = None
    
    if args.mode == 'train_unet':
        # 单独训练U-Net模型
        print("\n===== 正在训练U-Net模型 =====")
        unet_model, unet_history = train_unet_model(args, device, data_loaders)
    
    elif args.mode == 'train_rl':
        # 单独训练PPO强化学习代理
        print("\n===== 正在训练PPO强化学习代理 =====")
        rl_agent, rl_history = train_rl_agent(args, device, data_loaders)
        
    elif args.mode == 'train_simple_rl':
        # 单独训练InteractiveRL代理
        print("\n===== 正在训练InteractiveRL代理 =====")
        # 创建目录保存训练数据
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'results/simple_rl_{timestamp}'
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建训练参数
        simple_rl_args = argparse.Namespace(
            data_dir=args.data_dir,
            output_dir=output_dir,
            lr=args.lr,
            gamma=0.99,
            num_episodes=args.simple_rl_episodes,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            update_interval=5,
            log_interval=5,
            eval_interval=args.eval_interval,
            num_eval_episodes=args.num_eval_episodes,
            save_interval=10,
            device=device,
            seed=42
        )
        
        # 训练简单RL代理，这将保存训练历史数据
        print(f"开始训练InteractiveRL，保存训练数据到 {output_dir}")
        from code.simple_rl import train_agent
        training_results = train_agent(simple_rl_args)
        
        print("\n===== InteractiveRL训练完成 =====")
        print(f"训练历史保存至: {output_dir}")
        print(f"最佳验证Dice: {training_results['best_val_dice']:.4f} (Episode {training_results['best_episode']})")
        print(f"最佳验证IoU: {training_results['best_val_iou']:.4f}")
        
        # 创建图表目录
        plot_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # 生成训练图表
        print("正在生成InteractiveRL训练图表...")
        
        # 绘制训练曲线
        if 'training_history' in training_results:
            history = training_results['training_history']
            
            plt.figure(figsize=(15, 10))
            
            # 绘制Dice分数曲线
            plt.subplot(2, 2, 1)
            plt.plot(history['episodes'], history['train_dice_scores'], 'b-', label='Training Dice')
            if history['val_dice_scores']:
                val_episodes = history['episodes'][::simple_rl_args.eval_interval][:len(history['val_dice_scores'])]
                plt.plot(val_episodes, history['val_dice_scores'], 'r-', label='Validation Dice')
                plt.axhline(y=history['best_val_dice'], color='g', linestyle='--', 
                          label=f'Best Val Dice: {history["best_val_dice"]:.4f}')
            plt.title('Dice Score Progress')
            plt.xlabel('Episode')
            plt.ylabel('Dice Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 绘制奖励曲线
            plt.subplot(2, 2, 2)
            plt.plot(history['episodes'], history['train_rewards'], 'g-')
            plt.title('Training Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Episode Reward')
            plt.grid(True, alpha=0.3)
            
            # 绘制损失曲线
            plt.subplot(2, 2, 3)
            plt.plot(history['episodes'], history['policy_losses'], 'r-', label='Policy Loss')
            plt.plot(history['episodes'], history['value_losses'], 'b-', label='Value Loss')
            plt.title('Training Losses')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 绘制每个episode的步数
            plt.subplot(2, 2, 4)
            plt.plot(history['episodes'], history['steps_per_episode'], 'o-')
            plt.title('Steps per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'interactiverl_training_curves.png'), dpi=300)
            plt.close()
            
            print(f"训练图表已保存至: {plot_dir}")
    
    elif args.mode == 'evaluate':
        # 只评估模型，不训练
        print("\n===== 评估模型性能 =====")
        # 尝试加载现有模型
        try:
            # 尝试加载U-Net模型
            unet_model = UNet(n_channels=3, n_classes=1, bilinear=True)
            unet_model.load_state_dict(torch.load('models/unet_model.pth', map_location=device))
            unet_model = unet_model.to(device)
            print("成功加载U-Net模型")
        except Exception as e:
            print(f"无法加载U-Net模型: {str(e)}")
            unet_model = None
            
        results = evaluate_models(args, device, data_loaders, unet_model, rl_agent)
        
    elif args.mode == 'compare_unet_simple_rl':
        # 比较U-Net和SimpleRL
        print("\n===== 比较U-Net和SimpleRL性能 =====")
        results = compare_unet_simple_rl(args, device, data_loaders)
        
    elif args.mode == 'train_and_evaluate':
        # 训练并评估模型
        print("\n===== 训练并评估模型 =====")
        results, models_dir, results_dir = train_and_evaluate(args, device, data_loaders)
        print('\n===== 训练和评估完成 =====')
        print(f'结果保存至: {results_dir}')
        print(f'模型保存至: {models_dir}')
        
        # 如果同时有两个模型的评估结果，显示比较
        if 'unet' in results and 'simple_rl' in results:
            unet_dice = results['unet']['mean_dice']
            unet_iou = results['unet']['mean_iou']
            rl_dice = results['simple_rl']['mean_dice']
            rl_iou = results['simple_rl']['mean_iou']
            
            winner = "U-Net" if unet_dice > rl_dice else "SimpleRL"
            
            print(f'\n性能总结:')
            print(f'--------------------')
            print(f'U-Net:    Dice={unet_dice:.4f}, IoU={unet_iou:.4f}')
            print(f'SimpleRL: Dice={rl_dice:.4f}, IoU={rl_iou:.4f}')
            print(f'\n表现更好的模型: {winner} (基于Dice分数)')
            print(f'Dice分数差异: {abs(unet_dice - rl_dice):.4f}')
            
            # 显示可视化文件的位置
            print(f'\n可视化文件:')
            print(f' - 训练历史: {os.path.join(results_dir, "unet_training_history.png")}')
            print(f' - 性能对比: {os.path.join(results_dir, "performance_comparison.png")}')
            if args.save_visualizations:
                print(f' - 分割样例: {os.path.join(results_dir, "visualizations/")}')
    
    elif args.mode == 'all':
        # 默认模式改为仅训练和评估U-Net
        print("\n===== 仅训练和评估U-Net模型 =====")
        # 训练U-Net
        unet_model, unet_history = train_unet_model(args, device, data_loaders)
        
        # 评估U-Net
        test_results = evaluate_unet(
            model=unet_model,
            test_dataset=data_loaders['test'].dataset,
            device=device,
            save_dir='results/unet_evaluation',
            num_samples=args.eval_samples,
            save_visualizations=args.save_visualizations
        )
        
        print(f"\nU-Net测试结果:")
        print(f"平均Dice分数: {test_results['mean_dice']:.4f}")
        print(f"平均IoU分数: {test_results['mean_iou']:.4f}")
    
    print('完成!')

if __name__ == '__main__':
    main() 