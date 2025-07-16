# Hierarchical_PRM.py
"""
完整的分层预测表示模型，集成h-DQN论文中的ε-greedy退火算法
包含双层探索策略、成功率跟踪和动态退火机制
"""

from typing import List, Optional, Tuple, Dict
import torch
import torch.nn.functional as F
from torch import nn
import os
import math
import numpy as np
from torch.distributions import Categorical, Normal
import random
from collections import defaultdict, deque

# 导入原始PRM组件
try:
    from model.PRM import (
        Encoder, ImageDecoder, ActionEncoder, TransformerLayer,
        create_optimized_optimizer, create_lr_scheduler
    )
except ImportError:
    print("Warning: Could not import from model.PRM, using embedded components")


class OptionEncoder(nn.Module):
    """选项编码器，将选项ID编码为向量"""

    def __init__(self, num_options=4, embed_dim=128, output_dim=256):
        super().__init__()
        self.option_embedding = nn.Embedding(num_options, embed_dim)
        self.option_proj = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, options):
        """
        Args:
            options: (B,) 选项索引
        Returns:
            option_features: (B, output_dim) 选项特征
        """
        embedded = self.option_embedding(options.long())
        return self.option_proj(embedded)


class HighLevelPolicy(nn.Module):
    """高层策略网络 - 选择选项 (Meta-controller)"""

    def __init__(self, state_dim=256, num_options=4, hidden_dim=512):
        super().__init__()
        self.num_options = num_options

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_options)
        )

    def forward(self, state_features):
        """
        Args:
            state_features: (B, state_dim) 状态特征
        Returns:
            option_logits: (B, num_options) 选项logits
        """
        return self.policy_net(state_features)

    def sample_option(self, state_features):
        """采样选项"""
        logits = self.forward(state_features)
        dist = Categorical(logits=logits)
        option = dist.sample()
        log_prob = dist.log_prob(option)
        return option, log_prob, dist.entropy()


class OptionPolicy(nn.Module):
    """选项策略网络 - 在给定选项下选择动作 (Controller)"""

    def __init__(self, state_dim=256, option_dim=256, action_dim=19, hidden_dim=512):
        super().__init__()
        self.action_dim = action_dim

        # 融合状态和选项信息
        self.fusion_net = nn.Sequential(
            nn.Linear(state_dim + option_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # 动作概率输出
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, state_features, option_features):
        """
        Args:
            state_features: (B, state_dim) 状态特征
            option_features: (B, option_dim) 选项特征
        Returns:
            action_logits: (B, action_dim) 动作logits
        """
        fused = torch.cat([state_features, option_features], dim=-1)
        hidden = self.fusion_net(fused)
        action_logits = self.action_head(hidden)
        return action_logits

    def sample_action(self, state_features, option_features):
        """采样动作"""
        logits = self.forward(state_features, option_features)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist.entropy()


class TerminationNetwork(nn.Module):
    """终止网络 - 决定是否终止当前选项"""

    def __init__(self, state_dim=256, option_dim=256, hidden_dim=512):
        super().__init__()

        self.termination_net = nn.Sequential(
            nn.Linear(state_dim + option_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 终止概率 [0, 1]
        )

    def forward(self, state_features, option_features):
        """
        Args:
            state_features: (B, state_dim) 状态特征
            option_features: (B, option_dim) 选项特征
        Returns:
            termination_prob: (B, 1) 终止概率
        """
        fused = torch.cat([state_features, option_features], dim=-1)
        return self.termination_net(fused)

    def should_terminate(self, state_features, option_features):
        """判断是否应该终止"""
        prob = self.forward(state_features, option_features)
        return torch.bernoulli(prob)


class ValueNetwork(nn.Module):
    """价值网络 - 估计状态价值"""

    def __init__(self, state_dim=256, hidden_dim=512):
        super().__init__()

        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state_features):
        """
        Args:
            state_features: (B, state_dim) 状态特征
        Returns:
            values: (B, 1) 状态价值
        """
        return self.value_net(state_features)


class HierarchicalPRM(nn.Module):
    """分层预测表示模型"""

    def __init__(
            self,
            # PRM配置
            img_in_channels: int = 3,
            img_out_channels: int = 3,
            encoder_layers: List[int] = [2, 3, 4, 3],
            decoder_layers: List[int] = [2, 2, 2, 1],
            base_channels: int = 64,
            latent_dim: int = 256,

            # 分层RL配置
            num_options: int = 4,  # 选项数量
            action_dim: int = 19,  # 动作维度 (0-18)
            hidden_dim: int = 512,

            # Transformer配置
            num_attention_heads: int = 8,
            transformer_layers: int = 3,

            # 训练配置
            use_skip_connections: bool = True,
            loss_weights: Dict[str, float] = None,
            dropout: float = 0.1
    ):
        super().__init__()

        if loss_weights is None:
            loss_weights = {
                'img': 1.0,  # 图像重建损失
                'done': 1.0,  # done预测损失
                'vector': 1.0,  # 特征一致性损失
                'policy': 1.0,  # 策略损失
                'value': 0.5,  # 价值损失
                'entropy': 0.01,  # 熵正则化
                'termination': 0.1  # 终止损失
            }

        self.num_options = num_options
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.loss_weights = loss_weights

        # === PRM 组件 ===
        self.encoder = Encoder(img_in_channels, encoder_layers, base_channels)
        self.image_decoder = ImageDecoder(
            latent_dim, img_out_channels, decoder_layers,
            base_channels, use_skip_connections
        )

        # === 分层RL组件 ===
        self.option_encoder = OptionEncoder(num_options, 128, latent_dim)
        self.high_level_policy = HighLevelPolicy(latent_dim, num_options, hidden_dim)
        self.option_policy = OptionPolicy(latent_dim, latent_dim, action_dim, hidden_dim)
        self.termination_network = TerminationNetwork(latent_dim, latent_dim, hidden_dim)
        self.value_network = ValueNetwork(latent_dim, hidden_dim)

        # === Done分类器 ===
        self.done_classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """初始化权重"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.02)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def encode_state(self, images):
        """编码状态"""
        state_features, skip_features = self.encoder(images)
        return state_features, skip_features

    def predict_next_frame(self, state_features, skip_features=None):
        """预测下一帧图像"""
        if skip_features is not None:
            predicted_image = self.image_decoder(state_features, skip_features)
        else:
            predicted_image = self.image_decoder(state_features)
        return predicted_image

    def select_option(self, state_features):
        """选择选项"""
        return self.high_level_policy.sample_option(state_features)

    def select_action(self, state_features, option):
        """根据选项选择动作"""
        option_features = self.option_encoder(option)
        return self.option_policy.sample_action(state_features, option_features)

    def should_terminate_option(self, state_features, option):
        """判断是否终止选项"""
        option_features = self.option_encoder(option)
        return self.termination_network.should_terminate(state_features, option_features)

    def get_state_value(self, state_features):
        """获取状态价值"""
        return self.value_network(state_features)

    def forward(self, images, options=None, actions=None, return_all=False):
        """
        前向传播
        Args:
            images: (B, 3, H, W) 输入图像
            options: (B,) 选项 (可选)
            actions: (B,) 动作 (可选)
            return_all: 是否返回所有中间结果
        """
        batch_size = images.size(0)

        # 编码状态
        state_features, skip_features = self.encode_state(images)

        # 预测下一帧
        predicted_images = self.predict_next_frame(state_features, skip_features)

        # Done预测
        done_predictions = self.done_classifier(state_features)

        # 状态价值
        state_values = self.get_state_value(state_features)

        results = {
            'state_features': state_features,
            'predicted_images': predicted_images,
            'done_predictions': done_predictions,
            'state_values': state_values
        }

        # 如果提供了选项，计算策略相关输出
        if options is not None:
            option_features = self.option_encoder(options)

            # 动作概率
            action_logits = self.option_policy(state_features, option_features)
            results['action_logits'] = action_logits

            # 终止概率
            termination_probs = self.termination_network(state_features, option_features)
            results['termination_probs'] = termination_probs

        # 高层策略 - 选项选择
        option_logits = self.high_level_policy(state_features)
        results['option_logits'] = option_logits

        if return_all:
            results['skip_features'] = skip_features

        return results

    def compute_hierarchical_loss(self,
                                  batch_data: Dict[str, torch.Tensor],
                                  predictions: Dict[str, torch.Tensor],
                                  target_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算分层学习损失
        """
        losses = {}

        # 1. 图像重建损失
        if 'predicted_images' in predictions and 'target_images' in target_data:
            losses['img'] = F.mse_loss(
                predictions['predicted_images'],
                target_data['target_images']
            )

        # 2. Done预测损失
        if 'done_predictions' in predictions and 'target_done' in target_data:
            target_done_float = target_data['target_done'].float().view(-1, 1)
            losses['done'] = F.binary_cross_entropy(
                predictions['done_predictions'],
                target_done_float
            )

        # 3. 特征一致性损失
        if 'state_features' in predictions and 'target_features' in target_data:
            pred_norm = F.normalize(predictions['state_features'], p=2, dim=1)
            target_norm = F.normalize(target_data['target_features'], p=2, dim=1)
            cosine_sim = (pred_norm * target_norm).sum(dim=1)
            losses['vector'] = (1.0 - cosine_sim).mean()

        # 4. 策略损失 (如果有动作标签)
        if 'action_logits' in predictions and 'target_actions' in target_data:
            losses['policy'] = F.cross_entropy(
                predictions['action_logits'],
                target_data['target_actions']
            )

        # 5. 价值损失 (如果有价值标签)
        if 'state_values' in predictions and 'target_values' in target_data:
            losses['value'] = F.mse_loss(
                predictions['state_values'].squeeze(),
                target_data['target_values']
            )

        # 6. 熵正则化
        if 'option_logits' in predictions:
            option_dist = Categorical(logits=predictions['option_logits'])
            losses['entropy'] = -option_dist.entropy().mean()

        # 7. 终止损失 (如果有终止标签)
        if 'termination_probs' in predictions and 'target_termination' in target_data:
            target_term_float = target_data['target_termination'].float().view(-1, 1)
            losses['termination'] = F.binary_cross_entropy(
                predictions['termination_probs'],
                target_term_float
            )

        # 计算总损失
        total_loss = 0
        loss_dict = {}
        for loss_name, loss_value in losses.items():
            weight = self.loss_weights.get(loss_name, 1.0)
            weighted_loss = weight * loss_value
            total_loss += weighted_loss
            loss_dict[f'loss_{loss_name}'] = loss_value.item()
            loss_dict[f'weighted_loss_{loss_name}'] = weighted_loss.item()

        loss_dict['total_loss'] = total_loss.item()

        # 计算一些额外指标
        if 'done_predictions' in predictions and 'target_done' in target_data:
            done_pred_binary = (predictions['done_predictions'] > 0.5).float()
            done_accuracy = (done_pred_binary.view(-1) == target_data['target_done'].float()).float().mean()
            loss_dict['done_accuracy'] = done_accuracy.item()

        if 'action_logits' in predictions and 'target_actions' in target_data:
            action_pred = predictions['action_logits'].argmax(dim=1)
            action_accuracy = (action_pred == target_data['target_actions']).float().mean()
            loss_dict['action_accuracy'] = action_accuracy.item()

        return total_loss, loss_dict


# ============================================================================
# 🔥 ε-greedy退火算法组件 (h-DQN论文实现)
# ============================================================================

class EpsilonAnnealingScheduler:
    """
    🔥 ε-greedy退火调度器
    实现h-DQN论文中的探索概率退火机制

    Learning Algorithm中描述的退火策略：
    - Meta-controller: ε2从1退火到0.1
    - Controller: ε1,g基于成功率动态调整（成功率>90%时ε1,g=0.1，否则退火）
    """

    def __init__(self,
                 initial_epsilon_meta: float = 1.0,  # Meta-controller初始ε2
                 final_epsilon_meta: float = 0.1,  # Meta-controller最终ε2
                 initial_epsilon_controller: float = 1.0,  # Controller初始ε1
                 final_epsilon_controller: float = 0.1,  # Controller最终ε1
                 success_threshold: float = 0.9,  # 成功率阈值(90%)
                 anneal_steps_meta: int = 50000,  # Meta-controller退火步数
                 anneal_steps_controller: int = 100000):  # Controller退火步数

        self.initial_epsilon_meta = initial_epsilon_meta
        self.final_epsilon_meta = final_epsilon_meta
        self.initial_epsilon_controller = initial_epsilon_controller
        self.final_epsilon_controller = final_epsilon_controller
        self.success_threshold = success_threshold
        self.anneal_steps_meta = anneal_steps_meta
        self.anneal_steps_controller = anneal_steps_controller

        self.current_step = 0

        # 🔥 记录每个目标(选项)的成功率 - h-DQN Algorithm 3
        self.goal_success_rates = defaultdict(float)
        self.goal_attempts = defaultdict(int)
        self.goal_successes = defaultdict(int)

        print(f"🔥 h-DQN ε-greedy退火调度器初始化:")
        print(f"   Meta-controller: ε2 {initial_epsilon_meta} → {final_epsilon_meta} ({anneal_steps_meta}步)")
        print(
            f"   Controller: ε1 {initial_epsilon_controller} → {final_epsilon_controller} ({anneal_steps_controller}步)")
        print(f"   成功率阈值: {success_threshold}")

    def get_epsilon_meta(self) -> float:
        """
        🔥 获取Meta-controller的当前ε2值
        h-DQN论文: ε2从初始值1线性退火
        """
        if self.current_step >= self.anneal_steps_meta:
            return self.final_epsilon_meta

        progress = self.current_step / self.anneal_steps_meta
        epsilon = self.initial_epsilon_meta - (self.initial_epsilon_meta - self.final_epsilon_meta) * progress
        return max(epsilon, self.final_epsilon_meta)

    def get_epsilon_controller(self, goal_id: int) -> float:
        """
        🔥 获取Controller对特定目标的当前ε1,g值
        h-DQN论文: 如果成功率>90%，设置ε1,g=0.1，否则退火到0.1

        Args:
            goal_id: 目标(选项)ID

        Returns:
            epsilon_1_g: 该目标的探索概率
        """
        success_rate = self.goal_success_rates[goal_id]

        # 🔥 h-DQN核心逻辑：成功率>90%时强制设置ε1,g=0.1
        if success_rate > self.success_threshold:
            return self.final_epsilon_controller

        # 否则使用退火策略
        if self.current_step >= self.anneal_steps_controller:
            return self.final_epsilon_controller

        progress = self.current_step / self.anneal_steps_controller
        epsilon = self.initial_epsilon_controller - (
                    self.initial_epsilon_controller - self.final_epsilon_controller) * progress
        return max(epsilon, self.final_epsilon_controller)

    def update_goal_success(self, goal_id: int, success: bool):
        """
        🔥 更新目标成功率统计
        h-DQN论文中用于决定ε1,g的关键机制

        Args:
            goal_id: 目标ID
            success: 是否成功达成目标
        """
        self.goal_attempts[goal_id] += 1
        if success:
            self.goal_successes[goal_id] += 1

        # 计算成功率
        if self.goal_attempts[goal_id] > 0:
            self.goal_success_rates[goal_id] = self.goal_successes[goal_id] / self.goal_attempts[goal_id]

    def step(self):
        """推进一步，更新当前步数"""
        self.current_step += 1

    def get_current_stats(self) -> Dict:
        """获取当前统计信息"""
        return {
            'current_step': self.current_step,
            'epsilon_meta': self.get_epsilon_meta(),
            'goal_success_rates': dict(self.goal_success_rates),
            'goal_attempts': dict(self.goal_attempts),
            'goal_successes': dict(self.goal_successes)
        }


class GoalSuccessTracker:
    """
    🔥 目标成功跟踪器
    用于判断选项执行是否成功，支持h-DQN的成功率计算
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.goal_windows = defaultdict(lambda: deque(maxlen=window_size))
        self.goal_total_attempts = defaultdict(int)
        self.goal_total_successes = defaultdict(int)

    def add_result(self, goal_id: int, success: bool):
        """添加目标执行结果"""
        self.goal_windows[goal_id].append(1.0 if success else 0.0)
        self.goal_total_attempts[goal_id] += 1
        if success:
            self.goal_total_successes[goal_id] += 1

    def get_success_rate(self, goal_id: int) -> float:
        """获取目标的滑动窗口成功率"""
        if goal_id not in self.goal_windows or len(self.goal_windows[goal_id]) == 0:
            return 0.0
        return sum(self.goal_windows[goal_id]) / len(self.goal_windows[goal_id])

    def get_total_success_rate(self, goal_id: int) -> float:
        """获取目标的总体成功率"""
        if self.goal_total_attempts[goal_id] == 0:
            return 0.0
        return self.goal_total_successes[goal_id] / self.goal_total_attempts[goal_id]

    def get_all_success_rates(self) -> Dict[int, float]:
        """获取所有目标的成功率"""
        return {goal_id: self.get_success_rate(goal_id) for goal_id in self.goal_windows.keys()}


class HierarchicalAgent:
    """
    🔥 分层强化学习智能体
    实现h-DQN的双层ε-greedy策略和退火算法

    核心特性：
    - Meta-controller: ε2-greedy选项选择
    - Controller: ε1,g-greedy动作选择（基于成功率）
    - 动态退火机制
    """

    def __init__(self,
                 model: HierarchicalPRM,
                 epsilon_scheduler: EpsilonAnnealingScheduler,
                 success_tracker: GoalSuccessTracker,
                 num_options: int = 4,
                 max_option_length: int = 8):

        self.model = model
        self.epsilon_scheduler = epsilon_scheduler
        self.success_tracker = success_tracker
        self.num_options = num_options
        self.max_option_length = max_option_length

        # 当前状态
        self.current_option = None
        self.option_step_count = 0
        self.option_start_state = None
        self.option_start_step = 0

        print(f"🤖 h-DQN分层智能体初始化完成")
        print(f"   选项数量: {num_options}")
        print(f"   最大选项长度: {max_option_length}")

    def select_option(self, state_features: torch.Tensor, use_epsilon_greedy: bool = True) -> Tuple[int, bool]:
        """
        🔥 Meta-controller: 使用ε2-greedy策略选择选项
        h-DQN Algorithm 1 & 2

        Args:
            state_features: 状态特征
            use_epsilon_greedy: 是否使用ε-greedy策略

        Returns:
            option_id: 选择的选项ID
            is_random: 是否是随机选择
        """
        if not use_epsilon_greedy:
            # 贪婪策略
            with torch.no_grad():
                option_logits = self.model.high_level_policy(state_features)
                option_id = option_logits.argmax(dim=1).item()
            return option_id, False

        # 🔥 ε2-greedy策略 (h-DQN Meta-controller)
        epsilon_meta = self.epsilon_scheduler.get_epsilon_meta()

        if random.random() < epsilon_meta:
            # 随机探索
            option_id = random.randint(0, self.num_options - 1)
            is_random = True
        else:
            # 贪婪选择
            with torch.no_grad():
                option_logits = self.model.high_level_policy(state_features)
                option_id = option_logits.argmax(dim=1).item()
            is_random = False

        return option_id, is_random

    def select_action(self,
                      state_features: torch.Tensor,
                      option_id: int,
                      use_epsilon_greedy: bool = True) -> Tuple[int, bool]:
        """
        🔥 Controller: 使用ε1,g-greedy策略选择动作
        h-DQN Algorithm 1 & 2 - 基于目标成功率的动态探索

        Args:
            state_features: 状态特征
            option_id: 当前选项ID
            use_epsilon_greedy: 是否使用ε-greedy策略

        Returns:
            action_id: 选择的动作ID
            is_random: 是否是随机选择
        """
        if not use_epsilon_greedy:
            # 贪婪策略
            with torch.no_grad():
                option_tensor = torch.tensor([option_id], device=state_features.device)
                option_features = self.model.option_encoder(option_tensor)
                action_logits = self.model.option_policy(state_features, option_features)
                action_id = action_logits.argmax(dim=1).item()
            return action_id, False

        # 🔥 ε1,g-greedy策略 (h-DQN Controller)
        epsilon_controller = self.epsilon_scheduler.get_epsilon_controller(option_id)

        if random.random() < epsilon_controller:
            # 随机探索
            action_id = random.randint(0, 18)  # Atari动作空间0-18
            is_random = True
        else:
            # 贪婪选择
            with torch.no_grad():
                option_tensor = torch.tensor([option_id], device=state_features.device)
                option_features = self.model.option_encoder(option_tensor)
                action_logits = self.model.option_policy(state_features, option_features)
                action_id = action_logits.argmax(dim=1).item()
            is_random = False

        return action_id, is_random

    def should_terminate_option(self, state_features: torch.Tensor, option_id: int) -> bool:
        """
        判断是否应该终止当前选项
        结合网络预测和强制终止条件

        Args:
            state_features: 当前状态特征
            option_id: 当前选项ID

        Returns:
            should_terminate: 是否应该终止
        """
        # 强制终止条件：达到最大选项长度
        if self.option_step_count >= self.max_option_length:
            return True

        # 使用终止网络判断
        with torch.no_grad():
            option_tensor = torch.tensor([option_id], device=state_features.device)
            option_features = self.model.option_encoder(option_tensor)
            termination_prob = self.model.termination_network(state_features, option_features)
            should_terminate = torch.bernoulli(termination_prob).item() > 0.5

        return should_terminate

    def act(self, state_features: torch.Tensor, force_new_option: bool = False) -> Tuple[int, int, Dict]:
        """
        🔥 智能体完整的行动流程
        实现h-DQN的双层决策机制

        Args:
            state_features: 状态特征
            force_new_option: 是否强制选择新选项

        Returns:
            action_id: 选择的动作
            option_id: 当前选项
            info: 额外信息
        """
        info = {}

        # 检查是否需要选择新选项
        need_new_option = (
                force_new_option or
                self.current_option is None or
                self.should_terminate_option(state_features, self.current_option)
        )

        if need_new_option:
            # 🔥 如果有之前的选项，记录其执行结果并更新成功率
            if self.current_option is not None and self.option_start_state is not None:
                # 成功判断逻辑（可根据具体任务调整）
                # 这里使用达到最大长度作为成功的简单判断
                success = self.option_step_count >= self.max_option_length

                # 🔥 更新h-DQN成功率统计
                self.epsilon_scheduler.update_goal_success(self.current_option, success)
                self.success_tracker.add_result(self.current_option, success)

                info['option_terminated'] = True
                info['option_length'] = self.option_step_count
                info['option_success'] = success
                info['terminated_option'] = self.current_option

            # 🔥 Meta-controller: 选择新选项
            self.current_option, option_random = self.select_option(state_features)
            self.option_step_count = 0
            self.option_start_state = state_features.clone()
            self.option_start_step = self.epsilon_scheduler.current_step

            info['new_option_selected'] = True
            info['option_random'] = option_random

        # 🔥 Controller: 在当前选项下选择动作
        action_id, action_random = self.select_action(state_features, self.current_option)
        self.option_step_count += 1

        # 🔥 更新h-DQN调度器
        self.epsilon_scheduler.step()

        # 收集信息
        info.update({
            'current_option': self.current_option,
            'option_step_count': self.option_step_count,
            'action_random': action_random,
            'epsilon_controller': self.epsilon_scheduler.get_epsilon_controller(self.current_option),
            'epsilon_meta': self.epsilon_scheduler.get_epsilon_meta(),  # 🔥 确保总是存在
            'option_success_rate': self.success_tracker.get_success_rate(self.current_option),
            'scheduler_stats': self.epsilon_scheduler.get_current_stats()
        })

        return action_id, self.current_option, info


def create_hierarchical_prm_model(config: Dict = None):
    """创建分层PRM模型的工厂函数"""
    if config is None:
        config = {
            'num_options': 4,
            'action_dim': 19,
            'latent_dim': 256,
            'hidden_dim': 512,
            'loss_weights': {
                'img': 1.0,
                'done': 1.0,
                'vector': 1.0,
                'policy': 1.0,
                'value': 0.5,
                'entropy': 0.01,
                'termination': 0.1
            }
        }

    model = HierarchicalPRM(**config)
    return model


def create_enhanced_hierarchical_agent(model_config: Dict = None,
                                       annealing_config: Dict = None) -> HierarchicalAgent:
    """
    🔥 创建集成h-DQN退火算法的分层智能体

    Args:
        model_config: 模型配置
        annealing_config: 退火配置

    Returns:
        agent: 分层智能体
    """
    if model_config is None:
        model_config = {
            'num_options': 4,
            'action_dim': 19,
            'latent_dim': 256,
            'hidden_dim': 512
        }

    if annealing_config is None:
        annealing_config = {
            'initial_epsilon_meta': 1.0,  # h-DQN论文中ε2初始值
            'final_epsilon_meta': 0.1,  # h-DQN论文中ε2最终值
            'initial_epsilon_controller': 1.0,  # h-DQN论文中ε1初始值
            'final_epsilon_controller': 0.1,  # h-DQN论文中ε1最终值
            'success_threshold': 0.9,  # h-DQN论文中90%成功率阈值
            'anneal_steps_meta': 50000,  # Meta-controller退火步数
            'anneal_steps_controller': 100000  # Controller退火步数
        }

    # 创建模型
    model = create_hierarchical_prm_model(model_config)

    # 🔥 创建h-DQN调度器和跟踪器
    epsilon_scheduler = EpsilonAnnealingScheduler(**annealing_config)
    success_tracker = GoalSuccessTracker()

    # 🔥 创建分层智能体
    agent = HierarchicalAgent(
        model=model,
        epsilon_scheduler=epsilon_scheduler,
        success_tracker=success_tracker,
        num_options=model_config['num_options'],
        max_option_length=8
    )

    return agent


if __name__ == '__main__':
    print("=== 🔥 Hierarchical PRM with h-DQN ε-greedy Annealing Testing ===")

    # 测试配置
    BATCH_SIZE = 4
    IMG_C, IMG_H, IMG_W = 3, 210, 160
    NUM_OPTIONS = 4
    ACTION_DIM = 19

    # 创建测试数据
    test_images = torch.rand(BATCH_SIZE, IMG_C, IMG_H, IMG_W)
    test_options = torch.randint(0, NUM_OPTIONS, (BATCH_SIZE,))
    test_actions = torch.randint(0, ACTION_DIM, (BATCH_SIZE,))
    test_next_images = torch.rand(BATCH_SIZE, IMG_C, IMG_H, IMG_W)
    test_done = torch.randint(0, 2, (BATCH_SIZE,))

    # 创建原始模型测试
    print("\n=== 原始分层PRM测试 ===")
    model = create_hierarchical_prm_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")

    # 测试前向传播
    model.eval()
    with torch.no_grad():
        results = model(test_images, options=test_options, return_all=True)
        print(f"✅ 状态特征: {results['state_features'].shape}")
        print(f"✅ 预测图像: {results['predicted_images'].shape}")
        print(f"✅ 动作logits: {results['action_logits'].shape}")
        print(f"✅ 选项logits: {results['option_logits'].shape}")

    # 🔥 创建增强版智能体测试
    print("\n=== 🔥 h-DQN智能体测试 ===")
    agent = create_enhanced_hierarchical_agent()

    # 模拟训练过程，测试退火机制
    test_state_features = torch.rand(1, 256)

    print(f"\n🧪 测试h-DQN双层ε-greedy策略...")
    for step in range(15):
        action_id, option_id, info = agent.act(test_state_features)

        print(f"\nStep {step + 1}:")
        print(f"  动作: {action_id}, 选项: {option_id}")
        print(f"  ε2 (meta): {info['epsilon_meta']:.4f}")
        print(f"  ε1 (controller): {info['epsilon_controller']:.4f}")
        print(f"  选项步数: {info['option_step_count']}")
        print(f"  选项成功率: {info['option_success_rate']:.3f}")

        if info.get('new_option_selected', False):
            print(f"  🎯 Meta-controller选择新选项! (随机: {info.get('option_random', False)})")
        if info.get('option_terminated', False):
            print(f"  ✅ 选项终止! 长度: {info.get('option_length', 0)}, 成功: {info.get('option_success', False)}")

    # 显示最终统计
    final_stats = agent.epsilon_scheduler.get_current_stats()
    print(f"\n📊 h-DQN最终统计:")
    print(f"  调度器步数: {final_stats['current_step']}")
    print(f"  当前ε2: {final_stats['epsilon_meta']:.4f}")
    print(f"  目标成功率: {final_stats['goal_success_rates']}")
    print(f"  目标尝试次数: {final_stats['goal_attempts']}")

    # 测试损失计算
    print("\n=== 分层损失计算测试 ===")
    model.train()
    predictions = model(test_images, options=test_options)
    with torch.no_grad():
        target_state_features, _ = model.encode_state(test_next_images)

    target_data = {
        'target_images': test_next_images,
        'target_done': test_done,
        'target_features': target_state_features,
        'target_actions': test_actions,
        'target_values': torch.randn(BATCH_SIZE),
        'target_termination': torch.randint(0, 2, (BATCH_SIZE,))
    }

    loss, loss_dict = model.compute_hierarchical_loss({}, predictions, target_data)
    print(f"✅ 总损失: {loss.item():.6f}")
    for key, value in loss_dict.items():
        print(f"   {key}: {value:.6f}")

    print("\n🎉 所有测试通过! 集成h-DQN退火算法的分层PRM准备就绪!")