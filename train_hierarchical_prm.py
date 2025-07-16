# train_hierarchical_prm.py
"""
🔥 完整的分层强化学习PRM模型训练脚本
集成h-DQN论文中的ε-greedy退火算法
支持CPU和GPU训练，包含课程学习、选项记忆、双层探索策略等高级功能
"""

import argparse
import logging
import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import datetime
import csv
import colorama
from colorama import Fore, Back, Style
import sys
import numpy as np
from collections import defaultdict
import random

# Import from other files
try:
    from env.AtariDataLoder import create_atari_dataloaders
    from model.Hierarchical_PRM import (
        HierarchicalPRM, create_hierarchical_prm_model,
        EpsilonAnnealingScheduler, GoalSuccessTracker, HierarchicalAgent,
        create_enhanced_hierarchical_agent
    )
    from model.PRM import create_optimized_optimizer, create_lr_scheduler
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required files are in the correct directories")
    print("Required files:")
    print("  - Hierarchical_PRM.py")
    print("  - model/PRM.py")
    print("  - env/AtariDataLoder.py")
    sys.exit(1)

# 初始化colorama
colorama.init()

logger = logging.getLogger(__name__)


def set_random_seed(seed=42):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"{Fore.GREEN}✅ Random seed set to {seed}{Style.RESET_ALL}")


def print_hierarchical_loss_info(loss_dict, global_step, epoch, batch_idx, lr=None, loss_type="TRAIN"):
    """
    🔥 用彩色打印分层学习损失信息（增强版，包含退火统计）
    """
    color = Fore.RED if loss_type == "TRAIN" else Fore.BLUE
    if "ANNEALING" in loss_type:
        color = Fore.MAGENTA

    print(f"\n{color}{'=' * 80}")
    print(f"{color}🔥 {loss_type} HIERARCHICAL LOSS - Step {global_step}")
    print(f"{color}{'=' * 80}")
    print(f"{Fore.YELLOW}Epoch: {epoch + 1:3d} | Batch: {batch_idx + 1:4d} | Step: {global_step:6d}")
    if lr is not None:
        print(f"{Fore.CYAN}Learning Rate: {lr:.2e}")

    print(f"{color}┌─ MAIN LOSSES ────────────────────────────────────────────┐")
    print(f"{color}│ Total Loss:      {loss_dict.get('total_loss', 0):10.6f}                    │")
    print(f"{color}│ Image Loss:      {loss_dict.get('loss_img', 0):10.6f}                    │")
    print(f"{color}│ Done Loss:       {loss_dict.get('loss_done', 0):10.6f}                    │")
    print(f"{color}│ Vector Loss:     {loss_dict.get('loss_vector', 0):10.6f}                    │")
    print(f"{color}├─ HIERARCHICAL LOSSES ────────────────────────────────────┤")
    print(f"{color}│ Policy Loss:     {loss_dict.get('loss_policy', 0):10.6f}                    │")
    print(f"{color}│ Value Loss:      {loss_dict.get('loss_value', 0):10.6f}                    │")
    print(f"{color}│ Entropy Loss:    {loss_dict.get('loss_entropy', 0):10.6f}                    │")
    print(f"{color}│ Termination:     {loss_dict.get('loss_termination', 0):10.6f}                    │")
    print(f"{color}├─ METRICS ────────────────────────────────────────────────┤")
    if 'done_accuracy' in loss_dict:
        print(f"{color}│ Done Accuracy:   {loss_dict['done_accuracy']:10.6f}                    │")
    if 'action_accuracy' in loss_dict:
        print(f"{color}│ Action Accuracy: {loss_dict['action_accuracy']:10.6f}                    │")
    if 'psnr' in loss_dict:
        print(f"{color}│ PSNR:            {loss_dict['psnr']:10.2f} dB                │")

    # 🔥 新增：h-DQN退火统计
    if 'epsilon_meta' in loss_dict:
        print(f"{color}├─ h-DQN ANNEALING STATS ──────────────────────────────────┤")
        print(f"{color}│ ε2 (Meta):       {loss_dict['epsilon_meta']:10.6f}                    │")
        if 'epsilon_controller_avg' in loss_dict:
            print(f"{color}│ ε1 (Ctrl Avg):   {loss_dict['epsilon_controller_avg']:10.6f}                    │")
        if 'annealing_step' in loss_dict:
            print(f"{color}│ Annealing Step:  {loss_dict['annealing_step']:10.0f}                    │")

    print(f"{color}└──────────────────────────────────────────────────────────┘")
    print(f"{color}{'=' * 80}{Style.RESET_ALL}")


class HierarchicalTrainingConfig:
    """🔥 分层训练配置 - 集成h-DQN退火配置"""

    def __init__(self):
        # 选项配置
        self.num_options = 4
        self.max_option_length = 8  # 适合15步序列
        self.min_option_length = 2

        # 🔥 h-DQN ε-greedy退火配置
        self.use_epsilon_annealing = True  # 是否启用h-DQN退火
        self.initial_epsilon_meta = 1.0  # Meta-controller初始ε2
        self.final_epsilon_meta = 0.1  # Meta-controller最终ε2
        self.initial_epsilon_controller = 1.0  # Controller初始ε1
        self.final_epsilon_controller = 0.1  # Controller最终ε1
        self.success_threshold = 0.9  # 90%成功率阈值 (h-DQN论文)
        self.anneal_steps_meta = 50000  # Meta-controller退火步数
        self.anneal_steps_controller = 100000  # Controller退火步数

        # 经验回放配置
        self.use_option_memory = True
        self.option_memory_size = 1000

        # 训练策略
        self.use_curriculum = True  # 是否使用课程学习
        self.curriculum_epochs = 10  # 课程学习轮数

        # 损失权重调度
        self.dynamic_loss_weights = True
        self.initial_weights = {
            'img': 1.0,
            'done': 1.0,
            'vector': 1.0,
            'policy': 0.1,  # 开始时较低
            'value': 0.05,
            'entropy': 0.001,
            'termination': 0.05
        }
        self.final_weights = {
            'img': 1.0,
            'done': 1.0,
            'vector': 1.0,
            'policy': 1.0,  # 逐渐增加
            'value': 0.5,
            'entropy': 0.01,
            'termination': 0.1
        }


class OptionMemory:
    """选项经验存储"""

    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, experience):
        """存储经验"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """随机采样经验"""
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)


def update_loss_weights(config: HierarchicalTrainingConfig, current_epoch: int, total_epochs: int):
    """动态更新损失权重"""
    if not config.dynamic_loss_weights:
        return config.final_weights

    # 线性插值
    progress = min(1.0, current_epoch / max(1, config.curriculum_epochs))
    weights = {}

    for key in config.initial_weights:
        initial = config.initial_weights[key]
        final = config.final_weights[key]
        weights[key] = initial + (final - initial) * progress

    return weights


def create_hierarchical_batch_data_with_agent(batch, device, config: HierarchicalTrainingConfig,
                                              agent: HierarchicalAgent = None):
    """
    🔥 使用h-DQN智能体的ε-greedy策略创建批次数据

    Args:
        batch: 原始批次数据
        device: 计算设备
        config: 训练配置
        agent: h-DQN分层智能体

    Returns:
        hierarchical_batch: 包含选项轨迹的批次数据
    """
    obs = batch['observations'].to(device, non_blocking=True)
    actions = batch['actions'].to(device, non_blocking=True)
    done = batch['done'].to(device, non_blocking=True)

    B, T_plus_1, C, H, W = obs.shape
    T = T_plus_1 - 1

    if agent is not None and config.use_epsilon_annealing:
        # 🔥 使用h-DQN智能体策略生成选项轨迹
        batch_options = []
        batch_terminations = []
        batch_agent_info = []

        for b in range(B):
            options = []
            terminations = []
            agent_info = []

            # 重置智能体状态
            agent.current_option = None
            agent.option_step_count = 0

            for t in range(T):
                # 获取当前状态特征
                current_obs = obs[b, t:t + 1, ...]
                with torch.no_grad():
                    state_features, _ = agent.model.encode_state(current_obs)

                # 🔥 智能体使用h-DQN策略选择动作和选项
                action_id, option_id, info = agent.act(state_features)

                options.append(option_id)
                # 检查是否终止
                terminated = info.get('option_terminated', False)
                terminations.append(1 if terminated else 0)
                agent_info.append(info)

            batch_options.append(options)
            batch_terminations.append(terminations)
            batch_agent_info.append(agent_info)

        # 转换为tensor
        options_tensor = torch.tensor(batch_options, device=device, dtype=torch.long)
        terminations_tensor = torch.tensor(batch_terminations, device=device, dtype=torch.long)

    else:
        # 原有的随机选项生成逻辑
        batch_options = []
        batch_option_lengths = []
        batch_terminations = []

        for b in range(B):
            options = []
            option_lengths = []
            terminations = []

            t = 0
            while t < T:
                # 随机选择选项
                current_option = random.randint(0, config.num_options - 1)
                remaining_steps = T - t

                if remaining_steps <= config.min_option_length:
                    option_length = remaining_steps
                else:
                    max_length = min(config.max_option_length, remaining_steps)
                    option_length = random.randint(config.min_option_length, max_length)

                # 填充选项
                for step_in_option in range(option_length):
                    if t < T:
                        options.append(current_option)
                        option_lengths.append(option_length)
                        terminations.append(1 if step_in_option == option_length - 1 else 0)
                        t += 1

            # 确保长度匹配
            options = options[:T]
            option_lengths = option_lengths[:T]
            terminations = terminations[:T]

            while len(options) < T:
                options.append(options[-1] if options else 0)
                option_lengths.append(option_lengths[-1] if option_lengths else 1)
                terminations.append(1)

            batch_options.append(options)
            batch_option_lengths.append(option_lengths)
            batch_terminations.append(terminations)

        options_tensor = torch.tensor(batch_options, device=device, dtype=torch.long)
        terminations_tensor = torch.tensor(batch_terminations, device=device, dtype=torch.long)

    # 重塑数据
    input_images = obs[:, :-1, ...].reshape(B * T, C, H, W)
    input_actions = actions[:, :-1].reshape(B * T)
    target_images = obs[:, 1:, ...].reshape(B * T, C, H, W)
    target_done = done[:, 1:].reshape(B * T)
    input_options = options_tensor.reshape(B * T)
    target_terminations = terminations_tensor.reshape(B * T)

    hierarchical_batch = {
        'input_images': input_images,
        'input_actions': input_actions,
        'input_options': input_options,
        'target_images': target_images,
        'target_done': target_done,
        'target_actions': input_actions,  # 使用当前动作作为监督信号
        'target_terminations': target_terminations,
        'batch_size': B,
        'sequence_length': T
    }

    # 如果使用智能体，添加额外信息
    if agent is not None and config.use_epsilon_annealing:
        hierarchical_batch['agent_info'] = batch_agent_info

    return hierarchical_batch


def train_hierarchical_epoch(model, dataloader, optimizer, scheduler, device, args, epoch,
                             csv_writer, csv_file, config: HierarchicalTrainingConfig,
                             agent: HierarchicalAgent = None):
    """
    🔥 分层训练的epoch函数 - 集成h-DQN退火算法
    """
    model.train()
    if agent is not None and config.use_epsilon_annealing:
        progress_desc = f"🔥 h-DQN Training Epoch {epoch + 1} (ε-annealing)"
    else:
        progress_desc = f"Hierarchical Training Epoch {epoch + 1}"

    progress_bar = tqdm(dataloader, desc=progress_desc, leave=False)

    # 更新损失权重
    current_weights = update_loss_weights(config, epoch, args.epochs)
    model.loss_weights = current_weights

    # 创建选项记忆
    if config.use_option_memory:
        option_memory = OptionMemory(config.option_memory_size)

    batch_count = 0
    epoch_losses = defaultdict(list)
    epoch_annealing_stats = defaultdict(list)  # 🔥 记录退火统计

    for batch_idx, batch in enumerate(progress_bar):
        global_step = epoch * len(dataloader) + batch_idx
        batch_count += 1

        optimizer.zero_grad()

        try:
            # 🔥 使用h-DQN智能体策略创建分层批次数据
            hierarchical_batch = create_hierarchical_batch_data_with_agent(batch, device, config, agent)

            # 🔥 记录h-DQN退火统计
            if agent is not None and config.use_epsilon_annealing:
                current_stats = agent.epsilon_scheduler.get_current_stats()
                epoch_annealing_stats['epsilon_meta'].append(current_stats['epsilon_meta'])
                epoch_annealing_stats['scheduler_step'].append(current_stats['current_step'])

                # 记录每个选项的探索概率
                for option_id in range(config.num_options):
                    eps_controller = agent.epsilon_scheduler.get_epsilon_controller(option_id)
                    epoch_annealing_stats[f'epsilon_controller_{option_id}'].append(eps_controller)

                # 记录成功率
                success_rates = agent.success_tracker.get_all_success_rates()
                for option_id, rate in success_rates.items():
                    epoch_annealing_stats[f'success_rate_{option_id}'].append(rate)

            # 前向传播
            predictions = model(
                hierarchical_batch['input_images'],
                options=hierarchical_batch['input_options']
            )

            # 获取目标特征
            with torch.no_grad():
                target_features, _ = model.encode_state(hierarchical_batch['target_images'])

            # 构造目标数据
            target_data = {
                'target_images': hierarchical_batch['target_images'],
                'target_done': hierarchical_batch['target_done'],
                'target_features': target_features,
                'target_actions': hierarchical_batch['target_actions'],
                'target_termination': hierarchical_batch['target_terminations']
            }

            # 如果使用价值学习，添加价值目标
            if args.use_value_learning:
                rewards = torch.where(hierarchical_batch['target_done'] == 1, 1.0, 0.0)
                target_data['target_values'] = rewards

            # 计算损失
            loss, loss_dict = model.compute_hierarchical_loss({}, predictions, target_data)

            # 检查损失是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"{Fore.YELLOW}Warning: Invalid loss detected, skipping batch{Style.RESET_ALL}")
                continue

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            # 记录损失
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)

            # 存储选项经验
            if config.use_option_memory:
                experience = {
                    'state_features': predictions['state_features'].detach().cpu(),
                    'options': hierarchical_batch['input_options'].cpu(),
                    'actions': hierarchical_batch['input_actions'].cpu(),
                    'rewards': target_data.get('target_values',
                                               torch.zeros_like(hierarchical_batch['target_done'])).cpu(),
                    'terminations': hierarchical_batch['target_terminations'].cpu()
                }
                option_memory.push(experience)

        except Exception as e:
            print(f"{Fore.RED}Error in batch {batch_idx}: {e}{Style.RESET_ALL}")
            continue

        current_lr = optimizer.param_groups[0]['lr']

        # 🔥 增强进度条显示，包含h-DQN退火信息
        postfix_dict = {
            'total': f"{loss.item():.4f}",
            'img': f"{loss_dict.get('loss_img', 0):.4f}",
            'policy': f"{loss_dict.get('loss_policy', 0):.4f}",
            'value': f"{loss_dict.get('loss_value', 0):.4f}",
            'lr': f"{current_lr:.2e}"
        }

        # 🔥 添加h-DQN退火信息
        if agent is not None and config.use_epsilon_annealing:
            postfix_dict.update({
                'ε2': f"{agent.epsilon_scheduler.get_epsilon_meta():.3f}",
                'ε1_avg': f"{np.mean([agent.epsilon_scheduler.get_epsilon_controller(i) for i in range(config.num_options)]):.3f}",
                'step': f"{agent.epsilon_scheduler.current_step}"
            })

        progress_bar.set_postfix(postfix_dict)

        # 每64个batch进行评估和打印
        if batch_count % 64 == 0:
            # 🔥 增强损失信息显示，包含退火统计
            enhanced_loss_dict = loss_dict.copy()
            if agent is not None and config.use_epsilon_annealing:
                enhanced_loss_dict.update({
                    'epsilon_meta': agent.epsilon_scheduler.get_epsilon_meta(),
                    'epsilon_controller_avg': np.mean(
                        [agent.epsilon_scheduler.get_epsilon_controller(i) for i in range(config.num_options)]),
                    'annealing_step': agent.epsilon_scheduler.current_step
                })
                loss_type = "TRAIN+h-DQN-ANNEALING"
            else:
                loss_type = "TRAIN"

            print_hierarchical_loss_info(enhanced_loss_dict, global_step + 1, epoch, batch_idx, current_lr, loss_type)

            # 🔥 写入增强CSV，包含退火统计
            if csv_writer is not None and csv_file is not None:
                try:
                    log_data = [
                        global_step + 1,
                        "hierarchical_train_annealing" if agent else "hierarchical_train",
                        loss_dict.get('total_loss', 0),
                        loss_dict.get('loss_img', 0),
                        loss_dict.get('loss_done', 0),
                        loss_dict.get('loss_vector', 0),
                        loss_dict.get('loss_policy', 0),
                        loss_dict.get('loss_value', 0),
                        loss_dict.get('loss_entropy', 0),
                        loss_dict.get('loss_termination', 0),
                        loss_dict.get('done_accuracy', 0),
                        loss_dict.get('action_accuracy', 0),
                        current_lr,
                        json.dumps(current_weights),
                        # 🔥 h-DQN退火相关字段
                        agent.epsilon_scheduler.get_epsilon_meta() if agent else 0,
                        json.dumps([agent.epsilon_scheduler.get_epsilon_controller(i) for i in
                                    range(config.num_options)]) if agent else "[]",
                        agent.epsilon_scheduler.current_step if agent else 0,
                        json.dumps(agent.success_tracker.get_all_success_rates()) if agent else "{}"
                    ]
                    csv_writer.writerow(log_data)
                    csv_file.flush()
                except Exception as e:
                    print(f"{Fore.RED}CSV写入错误: {e}{Style.RESET_ALL}")

        # 常规日志记录
        elif (batch_idx + 1) % args.log_interval == 0 and csv_writer is not None:
            try:
                regular_log_data = [
                    global_step + 1,
                    "hierarchical_train_annealing" if agent else "hierarchical_train",
                    loss_dict.get('total_loss', 0),
                    loss_dict.get('loss_img', 0),
                    loss_dict.get('loss_done', 0),
                    loss_dict.get('loss_vector', 0),
                    loss_dict.get('loss_policy', 0),
                    loss_dict.get('loss_value', 0),
                    loss_dict.get('loss_entropy', 0),
                    loss_dict.get('loss_termination', 0),
                    loss_dict.get('done_accuracy', 0),
                    loss_dict.get('action_accuracy', 0),
                    current_lr,
                    json.dumps(current_weights),
                    # h-DQN退火字段
                    agent.epsilon_scheduler.get_epsilon_meta() if agent else 0,
                    json.dumps([agent.epsilon_scheduler.get_epsilon_controller(i) for i in
                                range(config.num_options)]) if agent else "[]",
                    agent.epsilon_scheduler.current_step if agent else 0,
                    json.dumps(agent.success_tracker.get_all_success_rates()) if agent else "{}"
                ]
                csv_writer.writerow(regular_log_data)
                csv_file.flush()
            except Exception as e:
                print(f"{Fore.RED}CSV写入错误: {e}{Style.RESET_ALL}")

    # 🔥 返回包含退火统计的结果
    avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
    avg_annealing_stats = {key: np.mean(values) if values else 0 for key, values in epoch_annealing_stats.items()}

    return avg_losses, avg_annealing_stats


def add_hierarchical_training_args(parser: argparse.ArgumentParser):
    """添加分层训练相关参数"""

    # === 基础配置 ===
    basic_group = parser.add_argument_group('Basic Configuration')
    basic_group.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                             help='Device to use for training')
    basic_group.add_argument('--seed', type=int, default=42,
                             help='Random seed for reproducibility')

    # === 数据路径和加载配置 ===
    data_group = parser.add_argument_group('Data Loading Configuration')
    data_group.add_argument('--data-dir', type=str, default='./Data/test_data1',
                            help='Directory containing PKL files')
    data_group.add_argument('--num-workers', type=int, default=2,
                            help='Number of data loading worker processes')

    # === 数据预处理配置 ===
    preprocess_group = parser.add_argument_group('Data Preprocessing Configuration')
    preprocess_group.add_argument('--img-height', type=int, default=210,
                                  help='Target image height')
    preprocess_group.add_argument('--img-width', type=int, default=160,
                                  help='Target image width')

    # === 序列和批次配置 ===
    sequence_group = parser.add_argument_group('Sequence and Batch Configuration')
    sequence_group.add_argument('--sequence-length', type=int, default=16,
                                help='Length of each training sequence')
    sequence_group.add_argument('--batch-size', type=int, default=4,
                                help='Number of sequences per batch')
    sequence_group.add_argument('--overlap-steps', type=int, default=8,
                                help='Number of steps to move forward when creating overlapping sequences')

    # === 数据集划分配置 ===
    split_group = parser.add_argument_group('Dataset Split Configuration')
    split_group.add_argument('--train-split', type=float, default=0.8,
                             help='Proportion of episodes for training')
    split_group.add_argument('--val-split', type=float, default=0.1,
                             help='Proportion of episodes for validation')

    # === 训练配置 ===
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--epochs', type=int, default=20,
                             help='Number of training epochs')
    train_group.add_argument('--output-dir', type=str, default='./Output_Hierarchical',
                             help='Base directory to save all training outputs')
    train_group.add_argument('--run-name', type=str, default='hierarchical_hdqn_test',
                             help='A specific name for this training run')
    train_group.add_argument('--log-interval', type=int, default=20,
                             help='How often (in batches) to log to CSV')

    # === 🔥 h-DQN分层RL配置 ===
    hierarchical_group = parser.add_argument_group('🔥 h-DQN Hierarchical RL Configuration')
    hierarchical_group.add_argument('--num-options', type=int, default=4,
                                    help='Number of options in the hierarchy')
    hierarchical_group.add_argument('--max-option-length', type=int, default=8,
                                    help='Maximum length of an option')
    hierarchical_group.add_argument('--min-option-length', type=int, default=2,
                                    help='Minimum length of an option')
    hierarchical_group.add_argument('--use-option-memory', action='store_true',
                                    help='Use option experience replay')
    hierarchical_group.add_argument('--option-memory-size', type=int, default=1000,
                                    help='Size of option memory buffer')
    hierarchical_group.add_argument('--use-curriculum', action='store_true',
                                    help='Use curriculum learning for loss weights')
    hierarchical_group.add_argument('--curriculum-epochs', type=int, default=5,
                                    help='Number of epochs for curriculum learning')
    hierarchical_group.add_argument('--use-value-learning', action='store_true',
                                    help='Enable value function learning')

    # === 🔥 h-DQN ε-greedy退火配置 ===
    annealing_group = parser.add_argument_group('🔥 h-DQN ε-greedy Annealing Configuration')
    annealing_group.add_argument('--enable-annealing', action='store_true', default=True,
                                 help='Enable h-DQN ε-greedy annealing mechanism')
    annealing_group.add_argument('--initial-epsilon-meta', type=float, default=1.0,
                                 help='Meta-controller initial ε2 (h-DQN)')
    annealing_group.add_argument('--final-epsilon-meta', type=float, default=0.1,
                                 help='Meta-controller final ε2 (h-DQN)')
    annealing_group.add_argument('--initial-epsilon-controller', type=float, default=1.0,
                                 help='Controller initial ε1 (h-DQN)')
    annealing_group.add_argument('--final-epsilon-controller', type=float, default=0.1,
                                 help='Controller final ε1 (h-DQN)')
    annealing_group.add_argument('--success-threshold', type=float, default=0.9,
                                 help='Success rate threshold for ε1,g=0.1 (h-DQN 90%)')
    annealing_group.add_argument('--anneal-steps-meta', type=int, default=50000,
                                 help='Annealing steps for meta-controller')
    annealing_group.add_argument('--anneal-steps-controller', type=int, default=100000,
                                 help='Annealing steps for controller')

    # === PRM 模型配置 ===
    model_group = parser.add_argument_group('PRM Model Configuration')
    model_group.add_argument('--latent-dim', type=int, default=256,
                             help='Dimension of the latent space')
    model_group.add_argument('--base-channels', type=int, default=64,
                             help='Base number of channels for convolutional layers')
    model_group.add_argument('--transformer-layers', type=int, default=3,
                             help='Number of layers in the Transformer predictor')
    model_group.add_argument('--num-attention-heads', type=int, default=8,
                             help='Number of attention heads in the Transformer')
    model_group.add_argument('--hidden-dim', type=int, default=512,
                             help='Hidden dimension for hierarchical components')
    model_group.add_argument('--no-skip-connections', action='store_true',
                             help='Disable skip connections in the ImageDecoder')

    # === 优化器和调度器配置 ===
    optim_group = parser.add_argument_group('Optimizer and Scheduler Configuration')
    optim_group.add_argument('--lr', type=float, default=5e-5,
                             help='Base learning rate')
    optim_group.add_argument('--weight-decay', type=float, default=1e-2,
                             help='Weight decay for the AdamW optimizer')
    optim_group.add_argument('--warmup-steps', type=int, default=200,
                             help='Number of warmup steps for the learning rate scheduler')
    optim_group.add_argument('--grad-clip-norm', type=float, default=1.0,
                             help='Maximum norm for gradient clipping')

    # === 调试和日志配置 ===
    debug_group = parser.add_argument_group('Debug and Logging Configuration')
    debug_group.add_argument('--log-level', type=str, default='INFO',
                             choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                             help='Logging level')
    debug_group.add_argument('--max-episodes', type=int, default=100,
                             help='Maximum number of episodes to use (for debugging)')


def main():
    """
    🔥 分层强化学习训练主函数 - 集成h-DQN ε-greedy退火算法
    """
    parser = argparse.ArgumentParser(description="🔥 Train Hierarchical PRM with h-DQN ε-greedy Annealing.")
    add_hierarchical_training_args(parser)
    args = parser.parse_args()

    # 设置随机种子
    set_random_seed(args.seed)

    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print(f"{Fore.YELLOW}CUDA not available, falling back to CPU{Style.RESET_ALL}")
        args.device = 'cpu'

    device = torch.device(args.device)
    print(f"{Fore.GREEN}Using device: {device}{Style.RESET_ALL}")

    # 设置日志
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # 创建输出目录
    run_path = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_path, exist_ok=True)

    log_file_path = os.path.join(run_path, 'hierarchical_training.log')
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()]
    )

    # 🔥 创建增强CSV文件，包含h-DQN退火字段
    csv_file_path = os.path.join(run_path, 'hierarchical_loss_log_hdqn.csv')
    csv_file = open(csv_file_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)

    # 🔥 增强CSV表头，包含h-DQN退火统计
    enhanced_csv_header = [
        'global_step', 'mode', 'total_loss', 'loss_img', 'loss_done',
        'loss_vector', 'loss_policy', 'loss_value', 'loss_entropy', 'loss_termination',
        'done_accuracy', 'action_accuracy', 'learning_rate', 'loss_weights',
        # 🔥 h-DQN退火字段
        'epsilon_meta', 'epsilon_controllers', 'annealing_step', 'success_rates'
    ]
    csv_writer.writerow(enhanced_csv_header)
    csv_file.flush()

    print(f"{Fore.GREEN}{'=' * 80}")
    print(f"{Fore.GREEN}🔥 Starting Hierarchical PRM Training with h-DQN ε-greedy Annealing")
    print(f"{Fore.GREEN}{'=' * 80}")
    print(f"{Fore.CYAN}Run name: {args.run_name}")
    print(f"{Fore.CYAN}Output dir: {run_path}")
    print(f"{Fore.CYAN}Device: {device}")
    print(f"{Fore.CYAN}Batch size: {args.batch_size}")
    print(f"{Fore.CYAN}Sequence length: {args.sequence_length}")
    print(f"{Fore.CYAN}Number of options: {args.num_options}")
    print(f"{Fore.CYAN}Max episodes: {args.max_episodes}")
    if args.enable_annealing:
        print(f"{Fore.MAGENTA}🔥 h-DQN ε-greedy Annealing: ENABLED")
        print(
            f"{Fore.MAGENTA}   Meta ε2: {args.initial_epsilon_meta} → {args.final_epsilon_meta} ({args.anneal_steps_meta} steps)")
        print(
            f"{Fore.MAGENTA}   Controller ε1: {args.initial_epsilon_controller} → {args.final_epsilon_controller} ({args.anneal_steps_controller} steps)")
        print(f"{Fore.MAGENTA}   Success threshold: {args.success_threshold}")
    else:
        print(f"{Fore.YELLOW}h-DQN ε-greedy Annealing: DISABLED")
    print(f"{Fore.GREEN}{'=' * 80}{Style.RESET_ALL}")

    # 保存配置
    config_dict = vars(args)
    config_dict['h_dqn_annealing_enabled'] = args.enable_annealing
    with open(os.path.join(run_path, 'hierarchical_hdqn_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)

    # 创建分层训练配置
    training_config = HierarchicalTrainingConfig()
    training_config.num_options = args.num_options
    training_config.max_option_length = args.max_option_length
    training_config.min_option_length = args.min_option_length
    training_config.use_option_memory = args.use_option_memory
    training_config.option_memory_size = args.option_memory_size
    training_config.use_curriculum = args.use_curriculum
    training_config.curriculum_epochs = args.curriculum_epochs
    # 🔥 h-DQN退火配置
    training_config.use_epsilon_annealing = args.enable_annealing
    training_config.initial_epsilon_meta = args.initial_epsilon_meta
    training_config.final_epsilon_meta = args.final_epsilon_meta
    training_config.initial_epsilon_controller = args.initial_epsilon_controller
    training_config.final_epsilon_controller = args.final_epsilon_controller
    training_config.success_threshold = args.success_threshold
    training_config.anneal_steps_meta = args.anneal_steps_meta
    training_config.anneal_steps_controller = args.anneal_steps_controller

    # 创建数据加载器
    print(f"{Fore.YELLOW}Creating data loaders...{Style.RESET_ALL}")
    try:
        train_loader, val_loader = create_atari_dataloaders(args)
    except Exception as e:
        print(f"{Fore.RED}Failed to create data loaders: {e}{Style.RESET_ALL}")
        return

    if train_loader is None:
        logger.error("Failed to create train loader. Exiting.")
        return

    print(f"{Fore.GREEN}✅ Created dataloaders - Train: {len(train_loader)}{Style.RESET_ALL}")
    if val_loader:
        print(f"{Fore.GREEN}✅ Validation loader: {len(val_loader)}{Style.RESET_ALL}")

    # 🔥 创建h-DQN增强版分层智能体
    agent = None
    if args.enable_annealing:
        print(f"{Fore.YELLOW}🔥 Creating h-DQN Enhanced Hierarchical Agent...{Style.RESET_ALL}")

        model_config = {
            'num_options': args.num_options,
            'action_dim': 19,  # Atari动作数
            'latent_dim': args.latent_dim,
            'hidden_dim': args.hidden_dim,
            'base_channels': args.base_channels,
            'transformer_layers': args.transformer_layers,
            'num_attention_heads': args.num_attention_heads,
            'use_skip_connections': not args.no_skip_connections,
            'loss_weights': training_config.initial_weights
        }

        annealing_config = {
            'initial_epsilon_meta': args.initial_epsilon_meta,
            'final_epsilon_meta': args.final_epsilon_meta,
            'initial_epsilon_controller': args.initial_epsilon_controller,
            'final_epsilon_controller': args.final_epsilon_controller,
            'success_threshold': args.success_threshold,
            'anneal_steps_meta': args.anneal_steps_meta,
            'anneal_steps_controller': args.anneal_steps_controller
        }

        try:
            agent = create_enhanced_hierarchical_agent(model_config, annealing_config)
            model = agent.model.to(device)
            print(f"{Fore.GREEN}🔥 ✅ h-DQN Enhanced Agent Created Successfully{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Failed to create h-DQN agent: {e}{Style.RESET_ALL}")
            return
    else:
        # 使用原始模型
        print(f"{Fore.YELLOW}Creating standard Hierarchical PRM model...{Style.RESET_ALL}")
        model_config = {
            'num_options': args.num_options,
            'action_dim': 19,
            'latent_dim': args.latent_dim,
            'hidden_dim': args.hidden_dim,
            'base_channels': args.base_channels,
            'transformer_layers': args.transformer_layers,
            'num_attention_heads': args.num_attention_heads,
            'use_skip_connections': not args.no_skip_connections,
            'loss_weights': training_config.initial_weights
        }

        try:
            model = create_hierarchical_prm_model(model_config).to(device)
            print(f"{Fore.GREEN}✅ Standard Model Created{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Failed to create model: {e}{Style.RESET_ALL}")
            return

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{Fore.CYAN}Total parameters: {total_params:,}")
    print(f"{Fore.CYAN}Trainable parameters: {trainable_params:,}{Style.RESET_ALL}")

    # 创建优化器和调度器
    try:
        optimizer = create_optimized_optimizer(model, base_lr=args.lr, weight_decay=args.weight_decay)
        max_steps = args.epochs * len(train_loader)
        scheduler = create_lr_scheduler(optimizer, warmup_steps=args.warmup_steps, max_steps=max_steps)
        print(f"{Fore.GREEN}✅ Optimizer and Scheduler Created{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Failed to create optimizer/scheduler: {e}{Style.RESET_ALL}")
        return

    # 🔥 训练循环
    print(f"{Fore.YELLOW}🔥 Starting h-DQN Enhanced Training...{Style.RESET_ALL}")
    best_loss = float('inf')

    try:
        for epoch in range(args.epochs):
            print(f"\n{Fore.MAGENTA}{'=' * 80}")
            if args.enable_annealing:
                print(f"{Fore.MAGENTA}🔥 h-DQN HIERARCHICAL EPOCH {epoch + 1}/{args.epochs}")
                # 显示当前退火状态
                if agent:
                    current_eps_meta = agent.epsilon_scheduler.get_epsilon_meta()
                    avg_eps_controller = np.mean(
                        [agent.epsilon_scheduler.get_epsilon_controller(i) for i in range(args.num_options)])
                    current_step = agent.epsilon_scheduler.current_step
                    print(
                        f"{Fore.MAGENTA}   Current ε2: {current_eps_meta:.4f} | Avg ε1: {avg_eps_controller:.4f} | Step: {current_step}")
            else:
                print(f"{Fore.MAGENTA}🎯 HIERARCHICAL EPOCH {epoch + 1}/{args.epochs} (standard)")
            print(f"{Fore.MAGENTA}{'=' * 80}{Style.RESET_ALL}")

            # 🔥 训练
            epoch_result = train_hierarchical_epoch(
                model, train_loader, optimizer, scheduler, device, args, epoch,
                csv_writer, csv_file, training_config, agent
            )

            if len(epoch_result) == 2:
                epoch_losses, annealing_stats = epoch_result
            else:
                epoch_losses = epoch_result
                annealing_stats = {}

            # 打印epoch总结
            print(f"\n{Fore.GREEN}📊 EPOCH {epoch + 1} SUMMARY:")
            print(f"{Fore.GREEN}Average Total Loss: {epoch_losses.get('total_loss', 0):.6f}")
            print(f"{Fore.GREEN}Average Policy Loss: {epoch_losses.get('loss_policy', 0):.6f}")
            print(f"{Fore.GREEN}Average Value Loss: {epoch_losses.get('loss_value', 0):.6f}")
            print(f"{Fore.GREEN}Done Accuracy: {epoch_losses.get('done_accuracy', 0):.4f}")
            print(f"{Fore.GREEN}Action Accuracy: {epoch_losses.get('action_accuracy', 0):.4f}{Style.RESET_ALL}")

            # 🔥 打印h-DQN退火统计
            if args.enable_annealing and annealing_stats:
                print(f"\n{Fore.BLUE}🔥 h-DQN ANNEALING STATS:")
                print(f"{Fore.BLUE}Average ε2 (Meta): {annealing_stats.get('epsilon_meta', 0):.4f}")
                print(f"{Fore.BLUE}Average Annealing Step: {annealing_stats.get('scheduler_step', 0):.0f}")
                for i in range(args.num_options):
                    eps_key = f'epsilon_controller_{i}'
                    success_key = f'success_rate_{i}'
                    eps_val = annealing_stats.get(eps_key, 0)
                    success_val = annealing_stats.get(success_key, 0)
                    print(f"{Fore.BLUE}Option {i}: ε1={eps_val:.4f}, Success Rate={success_val:.3f}")
                print(f"{Fore.BLUE}{Style.RESET_ALL}")

            # 保存模型
            current_loss = epoch_losses.get('total_loss', float('inf'))
            if current_loss < best_loss:
                best_loss = current_loss
                best_checkpoint_path = os.path.join(run_path, 'best_hierarchical_hdqn_model.pth')
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"{Fore.GREEN}💾 Best model saved! (Loss: {best_loss:.6f}){Style.RESET_ALL}")

            # 定期保存
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(run_path, f'hierarchical_hdqn_model_epoch_{epoch + 1}.pth')
                torch.save(model.state_dict(), checkpoint_path)

                # 🔥 保存智能体状态
                if agent:
                    agent_checkpoint_path = os.path.join(run_path, f'hdqn_agent_state_epoch_{epoch + 1}.pth')
                    torch.save({
                        'model_state': model.state_dict(),
                        'epsilon_scheduler_state': {
                            'current_step': agent.epsilon_scheduler.current_step,
                            'goal_success_rates': dict(agent.epsilon_scheduler.goal_success_rates),
                            'goal_attempts': dict(agent.epsilon_scheduler.goal_attempts),
                            'goal_successes': dict(agent.epsilon_scheduler.goal_successes)
                        },
                        'success_tracker_state': {
                            'goal_windows': {k: list(v) for k, v in agent.success_tracker.goal_windows.items()},
                            'goal_total_attempts': dict(agent.success_tracker.goal_total_attempts),
                            'goal_total_successes': dict(agent.success_tracker.goal_total_successes)
                        }
                    }, agent_checkpoint_path)

                print(f"{Fore.GREEN}💾 Epoch {epoch + 1} checkpoint saved!{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Training error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
    finally:
        # 保存最终模型
        try:
            final_checkpoint_path = os.path.join(run_path, 'final_hierarchical_hdqn_model.pth')
            torch.save(model.state_dict(), final_checkpoint_path)

            # 🔥 保存最终智能体状态
            if agent:
                final_agent_path = os.path.join(run_path, 'final_hdqn_agent_state.pth')
                torch.save({
                    'model_state': model.state_dict(),
                    'epsilon_scheduler_state': {
                        'current_step': agent.epsilon_scheduler.current_step,
                        'goal_success_rates': dict(agent.epsilon_scheduler.goal_success_rates),
                        'goal_attempts': dict(agent.epsilon_scheduler.goal_attempts),
                        'goal_successes': dict(agent.epsilon_scheduler.goal_successes)
                    },
                    'success_tracker_state': {
                        'goal_windows': {k: list(v) for k, v in agent.success_tracker.goal_windows.items()},
                        'goal_total_attempts': dict(agent.success_tracker.goal_total_attempts),
                        'goal_total_successes': dict(agent.success_tracker.goal_total_successes)
                    },
                    'training_config': vars(training_config),
                    'args': vars(args)
                }, final_agent_path)

            print(f"\n{Fore.GREEN}{'=' * 80}")
            print(f"{Fore.GREEN}🎉 🔥 h-DQN HIERARCHICAL TRAINING COMPLETED!")
            print(f"{Fore.GREEN}{'=' * 80}")
            print(f"{Fore.GREEN}Best loss: {best_loss:.6f}")
            print(f"{Fore.GREEN}Final model saved: {final_checkpoint_path}")
            if 'best_checkpoint_path' in locals():
                print(f"{Fore.GREEN}Best model saved: {best_checkpoint_path}")
            if agent:
                print(f"{Fore.GREEN}🔥 h-DQN Agent state saved: {final_agent_path}")
                # 显示最终退火统计
                final_stats = agent.epsilon_scheduler.get_current_stats()
                print(f"{Fore.GREEN}🔥 Final h-DQN Stats:")
                print(f"{Fore.GREEN}   Total annealing steps: {final_stats['current_step']}")
                print(f"{Fore.GREEN}   Final ε2: {final_stats['epsilon_meta']:.4f}")
                print(f"{Fore.GREEN}   Final goal success rates: {final_stats['goal_success_rates']}")
            print(f"{Fore.GREEN}Loss log saved: {csv_file_path}")
            print(f"{Fore.GREEN}{'=' * 80}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error saving final model: {e}{Style.RESET_ALL}")

        csv_file.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}🔥 h-DQN Training interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}🔥 h-DQN Training failed with error: {e}{Style.RESET_ALL}")
        import traceback

        traceback.print_exc()
        raise