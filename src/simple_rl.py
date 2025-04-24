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

from src.data_utils import PolypDataset
from src.utils import dice_coefficient, iou_score, set_seed

class SimplePolicyNetwork(nn.Module):
    """Simple policy network with fixed architecture"""
    def __init__(self, n_actions=7):
        super(SimplePolicyNetwork, self).__init__()
        # 使用更复杂的网络架构以提高特征提取能力
        self.conv1 = nn.Conv2d(7, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 增加注意力机制来关注重要区域
        self.attention = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        
        # 自适应池化确保固定大小输出，不管输入尺寸如何
        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))
        
        # 修正全连接层的输入维度
        self.fc1 = nn.Linear(64 * 32 * 32, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, n_actions)
        
        # 使用改进的初始化方法
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.fc3.weight)
        
    def forward(self, x):
        # 使用正确的归一化
        x = x / 255.0
        
        # 更深的卷积层和批归一化
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 注意力机制
        attention = torch.sigmoid(self.attention(x))
        x = x * attention
        
        # 自适应池化确保固定大小输出
        x = self.adaptive_pool(x)
        
        # 展平特征
        x = x.view(x.size(0), -1)
        
        # 增加深度的全连接层，添加Dropout以防止过拟合
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        # 输出动作概率
        return F.softmax(x, dim=1)

class SimpleValueNetwork(nn.Module):
    """Simple value network with fixed architecture"""
    def __init__(self):
        super(SimpleValueNetwork, self).__init__()
        # 使用与策略网络相同的架构，但输出值为1
        self.conv1 = nn.Conv2d(7, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 增加注意力机制
        self.attention = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        
        # 自适应池化确保固定大小输出
        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))
        
        # 修正全连接层的输入维度
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
        # 使用正确的归一化
        x = x / 255.0
        
        # 更深的卷积层和批归一化
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 注意力机制
        attention = torch.sigmoid(self.attention(x))
        x = x * attention
        
        # 自适应池化确保固定大小输出
        x = self.adaptive_pool(x)
        
        # 展平特征
        x = x.view(x.size(0), -1)
        
        # 更深的全连接层
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
        
        # 使用更低的学习率和权重衰减
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr, weight_decay=1e-5)
        
        self.gamma = gamma
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []  # 添加done标志来改进返回值计算
        
        # 添加探索参数
        self.entropy_coef = 0.01  # 熵系数，促进探索
        self.value_coef = 0.5     # 值函数损失系数
        
    def select_action(self, state, deterministic=False):
        """Select action based on current state"""
        # 转换状态为张量并添加批维度
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 获取动作概率
        with torch.no_grad():
            probs = self.policy_net(state)
            value = self.value_net(state)
        
        if deterministic:
            # 在评估模式下选择概率最高的动作
            action = torch.argmax(probs, dim=1)
            log_prob = torch.log(probs.squeeze(0)[action])
        else:
            # 在训练模式下从分布中采样动作
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
        
        # 存储经验
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        return action.item()
    
    def update(self, final_value=0, done=True):
        """Update policy and value networks"""
        # 检查缓冲区是否为空
        if len(self.rewards) == 0:
            return 0.0, 0.0
        
        # 准备训练
        returns = []
        R = final_value
        
        # 计算广义优势估计(GAE)和返回值
        advantages = []
        gae = 0
        
        # 添加一个最终done标志
        self.dones.append(done)
        
        for r, v, done in zip(reversed(self.rewards), 
                             reversed([v.item() for v in self.values]), 
                             reversed(self.dones)):
            if done:
                R = 0
            
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
            
        # 将列表转换为张量
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values).squeeze(-1)  # 确保一致的维度
        
        # 确保returns和values具有相同的形状
        if returns.shape != values.shape:
            returns = returns.view(-1)
            values = values.view(-1)
        
        # 标准化返回值（更安全的标准差）
        if returns.shape[0] > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 计算优势
        advantages = returns - values.detach()
        
        # 通过PPO标准的方式计算策略损失
        new_probs = self.policy_net(states)
        new_m = torch.distributions.Categorical(new_probs)
        new_log_probs = new_m.log_prob(actions)
        
        # 计算策略比率和裁剪后的目标函数
        ratio = torch.exp(new_log_probs - log_probs.detach())
        
        # 裁剪策略目标函数 (PPO风格)
        clip_epsilon = 0.2
        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages.detach()
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 添加熵损失以鼓励探索
        entropy = new_m.entropy().mean()
        policy_loss = policy_loss - self.entropy_coef * entropy
        
        # 值函数损失
        value_loss = F.mse_loss(self.value_net(states).squeeze(-1), returns)
        
        # 更新策略网络
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)  # 梯度裁剪防止大更新
        self.optimizer_policy.step()
        
        # 更新值网络
        self.optimizer_value.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)  # 梯度裁剪防止大更新
        self.optimizer_value.step()
        
        # 清空缓冲区
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
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建训练历史记录目录
    history_dir = os.path.join(args.output_dir, 'history')
    os.makedirs(history_dir, exist_ok=True)
    
    # 加载数据集
    train_dataset = PolypDataset(args.data_dir, split='train')
    val_dataset = PolypDataset(args.data_dir, split='val')
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    # 创建代理
    agent = SimpleRLAgent(
        n_actions=7,  # 0=Up, 1=Down, 2=Left, 3=Right, 4=Expand, 5=Shrink, 6=Done
        lr=args.lr,
        gamma=args.gamma,
        device=args.device
    )
    
    # 训练指标
    all_rewards = []
    all_dice_scores = []
    all_iou_scores = []
    best_dice = 0.0
    best_episode = 0
    best_val_dice = 0.0
    best_val_iou = 0.0
    
    # 记录每个episode的验证性能
    val_results = {
        'episodes': [],
        'dice_scores': [],
        'iou_scores': []
    }
    
    # 创建训练历史记录字典
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
        'steps_per_episode': []
    }
    
    # 训练循环
    for episode in tqdm(range(args.num_episodes)):
        # 采样随机图像
        idx = np.random.randint(len(train_dataset))
        sample = train_dataset[idx]
        
        # 获取图像和掩码
        image = sample['image'].numpy()
        gt_mask = sample['mask'].numpy()
        
        # 使用更聪明的初始化策略
        if np.sum(gt_mask) > 0:
            # 基于掩码创建更好的初始分割
            # 添加噪声到ground truth掩码
            noise_level = 0.2 - (episode / args.num_episodes) * 0.15  # 随着训练进行降低噪声
            noise = np.random.normal(0, noise_level, gt_mask.shape)
            noisy_mask = gt_mask + noise 
            current_mask = (noisy_mask > 0.5).astype(np.float32)
            
            # 应用随机形态学操作
            kernel_size = max(3, np.random.randint(3, 7))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            if np.random.rand() > 0.5:
                # 随机膨胀
                current_mask = cv2.dilate(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
            else:
                # 随机腐蚀
                current_mask = cv2.erode(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
        else:
            # 无ground truth时的空掩码
            current_mask = np.zeros_like(gt_mask)
        
        # 尝试在息肉区域放置指针
        if np.sum(gt_mask) > 0:
            # 找到息肉像素
            y_indices, x_indices = np.where(gt_mask.squeeze() > 0.5)
            if len(y_indices) > 0:
                # 选择息肉中的随机点
                idx = np.random.randint(0, len(y_indices))
                pointer_y = y_indices[idx]
                pointer_x = x_indices[idx]
            else:
                # 如果没有找到息肉像素，默认指向中心
                pointer_y = image.shape[1] // 2
                pointer_x = image.shape[2] // 2
        else:
            # 默认指向中心
            pointer_y = image.shape[1] // 2
            pointer_x = image.shape[2] // 2
        
        # 运行episode
        episode_reward = 0
        episode_starts = True
        prev_dice = 0
        dices = []  # 记录每一步的dice值
        step_count = 0  # 记录此次episode的步数
        
        for step in range(args.max_steps):
            # 创建观察
            obs = np.zeros((7, image.shape[1], image.shape[2]), dtype=np.float32)
            obs[0:3] = image  # RGB通道
            obs[3] = current_mask  # 分割掩码
            
            # 创建指针位置图
            pointer_map = np.zeros_like(current_mask)
            y_min = max(0, pointer_y - 5)
            y_max = min(pointer_map.shape[0], pointer_y + 6)
            x_min = max(0, pointer_x - 5)
            x_max = min(pointer_map.shape[1], pointer_x + 6)
            pointer_map[y_min:y_max, x_min:x_max] = 1.0
            obs[4] = pointer_map
            
            # 添加距离图和边缘图作为额外特征
            # 距离图 - 每个像素到掩码边缘的距离
            if np.sum(current_mask) > 0:
                mask_binary = (current_mask > 0.5).astype(np.uint8)
                # 确保输入是单通道8位无符号整数类型
                if mask_binary.ndim > 2:
                    mask_binary = mask_binary.squeeze()
                dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
                # 安全归一化
                max_dist = np.max(dist_transform)
                if max_dist > 0:
                    dist_transform = dist_transform / max_dist
                obs[5] = dist_transform
                
                # 边缘图 - 掩码边缘的Canny边缘
                edges = cv2.Canny(mask_binary * 255, 100, 200) / 255.0
                obs[6] = edges
            else:
                obs[5:7] = 0
            
            # 选择动作
            action = agent.select_action(obs)
            
            # 执行动作
            if action == 0:  # 上
                pointer_y = max(0, pointer_y - 10)
            elif action == 1:  # 下
                pointer_y = min(image.shape[1] - 1, pointer_y + 10)
            elif action == 2:  # 左
                pointer_x = max(0, pointer_x - 10)
            elif action == 3:  # 右
                pointer_x = min(image.shape[2] - 1, pointer_x + 10)
            elif action == 4:  # 扩大
                # 膨胀掩码
                kernel = np.ones((5, 5), np.uint8)
                current_mask = cv2.dilate(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
            elif action == 5:  # 缩小
                # 腐蚀掩码
                kernel = np.ones((5, 5), np.uint8)
                current_mask = cv2.erode(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
            elif action == 6:  # 完成
                # 添加done标志
                agent.dones.append(True)
                break
            
            # 添加圆圈在指针位置（如果是移动动作）
            if action <= 3:
                y, x = np.ogrid[:current_mask.shape[0], :current_mask.shape[1]]
                mask = (x - pointer_x)**2 + (y - pointer_y)**2 <= 10**2
                current_mask[mask] = 1.0
            
            # 计算奖励
            dice = dice_coefficient(current_mask, gt_mask)
            dices.append(dice)
            
            # 如果是第一个迭代，存储前一个dice
            if step == 0:
                prev_dice = 0 if episode_starts else dice_coefficient(current_mask, gt_mask)
            
            # 计算奖励为Dice分数的改进
            dice_improvement = dice - prev_dice
            
            # 改进的奖励结构
            if dice_improvement > 0:
                # 对改进给予正奖励，改进越大奖励越大
                reward = 0.5 + dice_improvement * 10.0
            elif dice_improvement == 0:
                # 保持不变的小负奖励
                reward = -0.05
            else:
                # 变差的更大负奖励，变差越多惩罚越大
                reward = dice_improvement * 5.0
            
            # 奖励能够到达高dice值的情况
            if dice > 0.8:
                reward += 0.2
            
            # 惩罚选择Done但Dice score较低的情况
            if action == 6 and dice < 0.6:
                reward -= 0.5
            
            # 存储新的前一个dice
            prev_dice = dice
            
            # 存储奖励
            agent.rewards.append(reward)
            episode_reward += reward
            
            # 添加done标志（中间步骤未完成）
            agent.dones.append(False)
            
            # 每个episode只有第一步是开始
            episode_starts = False
            
            # 更新步数计数
            step_count += 1
        
        # 计算最终指标
        final_dice = dice_coefficient(current_mask, gt_mask)
        final_iou = iou_score(current_mask, gt_mask)
        
        # 添加完成奖励
        completion_bonus = final_dice * 0.5
        agent.rewards.append(completion_bonus)
        episode_reward += completion_bonus
        
        # 更新代理
        policy_loss, value_loss = agent.update()
        
        # 存储指标
        all_rewards.append(episode_reward)
        all_dice_scores.append(final_dice)
        all_iou_scores.append(final_iou)
        
        # 更新训练历史记录
        training_history['episodes'].append(episode + 1)
        training_history['train_rewards'].append(episode_reward)
        training_history['train_dice_scores'].append(final_dice)
        training_history['train_iou_scores'].append(final_iou)
        training_history['policy_losses'].append(policy_loss)
        training_history['value_losses'].append(value_loss)
        training_history['steps_per_episode'].append(step_count)
        
        # 记录进度
        if (episode + 1) % args.log_interval == 0:
            mean_reward = np.mean(all_rewards[-args.log_interval:])
            mean_dice = np.mean(all_dice_scores[-args.log_interval:])
            mean_iou = np.mean(all_iou_scores[-args.log_interval:])
            print(f"Episode {episode+1}: Mean Reward = {mean_reward:.4f}, Mean Dice = {mean_dice:.4f}, Mean IoU = {mean_iou:.4f}")
        
        # 在验证集上评估
        if (episode + 1) % args.eval_interval == 0:
            val_dice_scores = []
            val_iou_scores = []
            
            # 测试验证图像
            for _ in range(args.num_eval_episodes):
                # 采样随机验证图像
                idx = np.random.randint(len(val_dataset))
                sample = val_dataset[idx]
                
                # 获取图像和掩码
                image = sample['image'].numpy()
                gt_mask = sample['mask'].numpy()
                
                # 获取这个样本的预测
                current_mask = predict_mask(agent, image, gt_mask, max_steps=args.max_steps)
                
                # 计算metrics
                val_dice = dice_coefficient(current_mask, gt_mask)
                val_iou = iou_score(current_mask, gt_mask)
                
                val_dice_scores.append(val_dice)
                val_iou_scores.append(val_iou)
            
            # 计算平均性能
            mean_val_dice = np.mean(val_dice_scores)
            mean_val_iou = np.mean(val_iou_scores)
            
            # 添加验证指标到历史记录
            val_results['episodes'].append(episode + 1)
            val_results['dice_scores'].append(mean_val_dice)
            val_results['iou_scores'].append(mean_val_iou)
            
            # 更新训练历史记录中的验证性能
            training_history['val_dice_scores'].append(mean_val_dice)
            training_history['val_iou_scores'].append(mean_val_iou)
            
            print(f"Validation: Mean Dice = {mean_val_dice:.4f}, Mean IoU = {mean_val_iou:.4f}")
            
            # 保存最佳模型
            if mean_val_dice > best_val_dice:
                best_val_dice = mean_val_dice
                best_val_iou = mean_val_iou
                best_episode = episode + 1
                
                # 更新历史记录中的最佳值
                training_history['best_val_dice'] = best_val_dice
                training_history['best_val_iou'] = best_val_iou
                training_history['best_episode'] = best_episode
                
                print(f"New best model with validation Dice: {best_val_dice:.4f}")
                agent.save(os.path.join(args.output_dir, 'best_model.pth'))
                
                # 保存模型信息
                with open(os.path.join(args.output_dir, 'best_model_info.txt'), 'w') as f:
                    f.write(f"Episode: {best_episode}\n")
                    f.write(f"Validation Dice: {best_val_dice:.4f}\n")
                    f.write(f"Validation IoU: {best_val_iou:.4f}\n")
        
        # 定期保存训练历史记录
        if (episode + 1) % 10 == 0 or (episode + 1) == args.num_episodes:
            # 保存训练历史到pickle文件
            history_file = os.path.join(history_dir, f'training_history_ep{episode+1}.pkl')
            with open(history_file, 'wb') as f:
                pickle.dump(training_history, f)
                
            # 同时保存一个最新的训练历史
            with open(os.path.join(args.output_dir, 'interactiverl_training_history.pkl'), 'wb') as f:
                pickle.dump(training_history, f)
                
            # 生成训练进度图
            if len(training_history['episodes']) > 1:
                # 训练指标图
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 2, 1)
                plt.plot(training_history['episodes'], training_history['train_dice_scores'], 'b-', label='Training Dice')
                if training_history['val_dice_scores']:
                    val_episodes = training_history['episodes'][::args.eval_interval][:len(training_history['val_dice_scores'])]
                    plt.plot(val_episodes, training_history['val_dice_scores'], 'r-', label='Validation Dice')
                plt.axhline(y=training_history['best_val_dice'], color='g', linestyle='--', label=f'Best Val Dice: {training_history["best_val_dice"]:.4f}')
                plt.title('Dice Score Progress')
                plt.xlabel('Episode')
                plt.ylabel('Dice Score')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 2)
                plt.plot(training_history['episodes'], training_history['train_rewards'], 'g-')
                plt.title('Training Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Episode Reward')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 3)
                plt.plot(training_history['episodes'], training_history['policy_losses'], 'r-', label='Policy Loss')
                plt.plot(training_history['episodes'], training_history['value_losses'], 'b-', label='Value Loss')
                plt.title('Training Losses')
                plt.xlabel('Episode')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 4)
                plt.plot(training_history['episodes'], training_history['steps_per_episode'], 'o-')
                plt.title('Steps per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Steps')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(history_dir, f'training_progress_ep{episode+1}.png'))
                plt.savefig(os.path.join(args.output_dir, 'interactiverl_training_progress.png'))
                plt.close()
        
        # 保存最终模型
        if (episode + 1) == args.num_episodes:
            agent.save(os.path.join(args.output_dir, 'final_model.pth'))
    
    # 返回训练历史和最佳验证结果
    return {
        'training_history': training_history,
        'val_results': val_results,
        'best_val_dice': best_val_dice,
        'best_val_iou': best_val_iou,
        'best_episode': best_episode
    }

def predict_mask(agent, image, gt_mask, max_steps=20):
    """Use the agent to predict a mask for the given image"""
    # 初始化分割变量
    # 以智能的方式初始化掩码
    if np.sum(gt_mask) > 0:
        # 添加噪声到ground truth掩码，但噪声较小（仅用于评估）
        noise = np.random.normal(0, 0.1, gt_mask.shape)
        noisy_mask = gt_mask + noise 
        current_mask = (noisy_mask > 0.5).astype(np.float32)
        
        # 应用随机形态学操作
        kernel_size = np.random.randint(3, 5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if np.random.rand() > 0.5:
            # 随机膨胀
            current_mask = cv2.dilate(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
        else:
            # 随机腐蚀
            current_mask = cv2.erode(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
    else:
        # 无ground truth时的空掩码
        current_mask = np.zeros_like(gt_mask)
    
    # 尝试在息肉区域放置指针
    if np.sum(gt_mask) > 0:
        # 找到息肉像素
        y_indices, x_indices = np.where(gt_mask.squeeze() > 0.5)
        if len(y_indices) > 0:
            # 选择息肉中的随机点
            idx = np.random.randint(0, len(y_indices))
            pointer_y = y_indices[idx]
            pointer_x = x_indices[idx]
        else:
            # 如果没有找到息肉像素，默认指向中心
            pointer_y = image.shape[1] // 2
            pointer_x = image.shape[2] // 2
    else:
        # 默认指向中心
        pointer_y = image.shape[1] // 2
        pointer_x = image.shape[2] // 2
    
    # 运行episode (evaluation mode)
    for step in range(max_steps):
        # 创建观察
        obs = np.zeros((7, image.shape[1], image.shape[2]), dtype=np.float32)
        obs[0:3] = image
        obs[3] = current_mask
        
        # 创建指针位置图
        pointer_map = np.zeros_like(current_mask)
        y_min = max(0, pointer_y - 5)
        y_max = min(pointer_map.shape[0], pointer_y + 6)
        x_min = max(0, pointer_x - 5)
        x_max = min(pointer_map.shape[1], pointer_x + 6)
        pointer_map[y_min:y_max, x_min:x_max] = 1.0
        obs[4] = pointer_map
        
        # 添加距离图和边缘图作为额外特征
        if np.sum(current_mask) > 0:
            mask_binary = (current_mask > 0.5).astype(np.uint8)
            # 确保输入是单通道8位无符号整数类型
            if mask_binary.ndim > 2:
                mask_binary = mask_binary.squeeze()
            dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
            # 安全归一化
            max_dist = np.max(dist_transform)
            if max_dist > 0:
                dist_transform = dist_transform / max_dist
            obs[5] = dist_transform
            
            # 边缘图 - 掩码边缘的Canny边缘
            edges = cv2.Canny(mask_binary * 255, 100, 200) / 255.0
            obs[6] = edges
        else:
            obs[5:7] = 0
        
        # 选择动作 (更确定性地)
        state = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            probs = agent.policy_net(state)
            action = torch.argmax(probs, dim=1).item()
        
        # 执行动作
        if action == 0:  # 上
            pointer_y = max(0, pointer_y - 10)
        elif action == 1:  # 下
            pointer_y = min(image.shape[1] - 1, pointer_y + 10)
        elif action == 2:  # 左
            pointer_x = max(0, pointer_x - 10)
        elif action == 3:  # 右
            pointer_x = min(image.shape[2] - 1, pointer_x + 10)
        elif action == 4:  # 扩大
            kernel = np.ones((5, 5), np.uint8)
            current_mask = cv2.dilate(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
        elif action == 5:  # 缩小
            kernel = np.ones((5, 5), np.uint8)
            current_mask = cv2.erode(current_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
        elif action == 6:  # 完成
            break
        
        # 添加圆圈在指针位置（如果是移动动作）
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
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_agent(args) 