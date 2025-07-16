# PRM_JE_Hierarchical.py - 分层版本的PRM_JE
# 基于原PRM_JE.py修改，新增分层强化学习功能

from typing import List, Optional, Tuple, Dict
import torch
import torch.nn.functional as F
from torch import nn
import os
import math
import numpy as np


# ============================================================================
# 基础组件定义
# ============================================================================

# SE注意力模块
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


# 增强残差块
class EnhancedResidualBlock(nn.Module):
    """增强的残差块，包含SE注意力"""

    def __init__(self, channels, num_layers=2, use_se=True):
        super().__init__()
        self.use_se = use_se

        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            ])
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))

        self.conv_layers = nn.Sequential(*layers)

        if self.use_se:
            self.se = SEBlock(channels)

        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv_layers(x)

        if self.use_se:
            out = self.se(out)

        out = out + identity
        out = self.final_relu(out)
        return out


class DownBlock2d(nn.Module):
    """下采样块"""

    def __init__(self, in_channels, out_channels, num_layers, downsample=False, use_se=True):
        super().__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.residual_blocks = nn.ModuleList([
            EnhancedResidualBlock(out_channels, 2, use_se) for _ in range(num_layers)
        ])

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        if self.downsample and identity.shape[2:] != out.shape[2:]:
            identity = F.adaptive_avg_pool2d(identity, out.shape[2:])

        out = out + identity

        for res_block in self.residual_blocks:
            out = res_block(out)

        return out


# 动作编码器
class ActionEncoder(nn.Module):
    """动作编码器，将0-18的整数动作编码为向量"""

    def __init__(self, action_dim=19, embed_dim=128, output_dim=256):
        super().__init__()
        self.action_embedding = nn.Embedding(action_dim, embed_dim)
        self.action_proj = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, actions):
        embedded = self.action_embedding(actions.long())
        return self.action_proj(embedded)


# 编码器
class Encoder(nn.Module):
    """编码器，输入210*160*3 图像"""

    def __init__(self, in_channels: int = 3, layers: List[int] = [2, 3, 4, 3], base_channels: int = 64) -> None:
        super().__init__()

        # Stem层 - 处理210*160*3输入
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # 特征提取网络
        self.layer1 = DownBlock2d(base_channels, base_channels, layers[0], downsample=False)
        self.layer2 = DownBlock2d(base_channels, base_channels * 2, layers[1], downsample=True)
        self.layer3 = DownBlock2d(base_channels * 2, base_channels * 4, layers[2], downsample=True)
        self.layer4 = DownBlock2d(base_channels * 4, base_channels * 8, layers[3], downsample=True)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 特征投影
        self.feature_proj = nn.Sequential(
            nn.Linear(base_channels * 8, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2:] == (210, 160), f"Expected input size (210, 160), got {x.shape[-2:]}"

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        pooled = self.global_pool(x)
        flattened = pooled.view(pooled.size(0), -1)
        features = self.feature_proj(flattened)

        return features


# ============================================================================
# 分层强化学习组件
# ============================================================================

# 目标编码器
class GoalEncoder(nn.Module):
    """目标编码器 - 将不同类型的目标编码为统一向量
    """

    def __init__(self, goal_dim: int = 64, output_dim: int = 256):
        super().__init__()
        # goal_dim:中间隐藏层维度,output_dim: 最终编码输出的统一维度

        # 空间目标编码器（x, y坐标）
        self.spatial_encoder = nn.Sequential(
            nn.Linear(2, goal_dim),
            nn.ReLU(),
            nn.Linear(goal_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        # 对象目标编码器（对象ID）
        # 假设最多10种对象
        self.object_encoder = nn.Sequential(
            nn.Embedding(10, goal_dim),
            nn.Linear(goal_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        # 探索目标编码器（探索方向）
        self.exploration_encoder = nn.Sequential(
            nn.Linear(4, goal_dim),  # 上下左右四个方向
            nn.ReLU(),
            nn.Linear(goal_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, goal_type: str, goal_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            goal_type: 目标类型 ('spatial', 'object', 'exploration')
            goal_params: 目标参数
        Returns:
            goal_features: 编码后的目标特征 (B, output_dim)
        """
        if goal_type == 'spatial':
            return self.spatial_encoder(goal_params)
        elif goal_type == 'object':
            return self.object_encoder(goal_params.long().squeeze(-1))
        elif goal_type == 'exploration':
            return self.exploration_encoder(goal_params)
        else:
            raise ValueError(f"Unknown goal type: {goal_type}")


# 元控制器
class MetaController(nn.Module):
    """元控制器 - 负责选择高级目标
    通过Q-network 评估各类目标的价值，输出 Q 值"""

    def __init__(self, state_dim: int = 256, hidden_dim: int = 512, num_goal_types: int = 3):
        super().__init__()

        # Q网络：估计选择不同目标类型的价值
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_goal_types)
        )

        # 目标参数生成器
        self.goal_param_generators = nn.ModuleDict({
            'spatial': nn.Sequential(
                nn.Linear(state_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2),  # x, y坐标
                nn.Tanh()  # 归一化到[-1, 1]
            ),
            'object': nn.Sequential(
                nn.Linear(state_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),  # 对象ID
                nn.Sigmoid()  # 归一化到[0, 1]，然后可以乘以对象数量
            ),
            'exploration': nn.Sequential(
                nn.Linear(state_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 4),  # 四个方向的权重
                nn.Softmax(dim=-1)
            )
        })

        self.goal_types = ['spatial', 'object', 'exploration']

    def forward(self, state_features: torch.Tensor, goal_type: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state_features: 状态特征 (B, state_dim)
            goal_type: 指定的目标类型，如果为None则返回所有类型的Q值
        Returns:
            goal_q_values: 目标选择Q值 (B, num_goal_types)
            goal_params: 目标参数 (B, param_dim) 或 None
        """
        # 计算目标选择Q值
        goal_q_values = self.q_network(state_features)

        if goal_type is not None:
            # 生成特定类型目标的参数
            goal_params = self.goal_param_generators[goal_type](state_features)
            return goal_q_values, goal_params

        return goal_q_values, None


# 内在奖励计算器
class IntrinsicCritic(nn.Module):
    """
    内在奖励计算器 - 评估目标完成情况/模型自己评估当前动作对实现高层目标有多大帮助
    接收当前状态特征和目标特征，输出一个 0 到 1 之间的值，表示完成目标的进展。值越高表示进展越好
    """

    def __init__(self, state_dim: int = 256, goal_dim: int = 256):
        super().__init__()

        self.reward_network = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出0-1的奖励
        )

    def forward(self, state_features: torch.Tensor, goal_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_features: 状态特征 (B, state_dim)
            goal_features: 目标特征 (B, goal_dim)
        Returns:
            intrinsic_reward: 内在奖励 (B, 1)
        """
        combined = torch.cat([state_features, goal_features], dim=-1)
        return self.reward_network(combined)


# 分层Transformer层
class HierarchicalTransformerLayer(nn.Module):
    """分层Transformer层，支持目标条件的注意力
    查询（Query）来自目标-动作融合特征，把高层目标和低层动作的信息结合在一起形成向量
    键（Key）和值（Value）来自视觉特征"""

    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # Goal-conditioned attention
        self.goal_q_proj = nn.Linear(d_model, d_model, bias=False)

        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x, goal_action_features):
        """
        Args:
            x: 视觉特征 (B, d_model)
            goal_action_features: 目标-动作融合特征 (B, d_model)
        """
        B = x.size(0)

        # 1. Goal-conditioned multi-head attention
        residual = x
        x_norm = self.ln1(x)
        goal_norm = self.ln1(goal_action_features)

        # 目标作为query, 视觉特征作为key和value
        Q = self.goal_q_proj(goal_norm).view(B, self.n_heads, self.head_dim)
        K = self.k_proj(x_norm).view(B, self.n_heads, self.head_dim)
        V = self.v_proj(x_norm).view(B, self.n_heads, self.head_dim)

        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attended = torch.matmul(attn_weights, V)
        attended = attended.view(B, self.d_model)
        attended = self.out_proj(attended)
        x = residual + attended

        # 2. Feed forward with residual connection
        residual = x
        x_norm = self.ln2(x)
        x = residual + self.ffn_dropout(self.ffn(x_norm))

        return x


# 分层预测模型
class HierarchicalPredictiveModel(nn.Module):
    """分层预测模型 - 集成目标信息"""

    def __init__(self, d_model=256, goal_dim=256, action_dim=19, n_heads=8, n_layers=3, d_ff=1024, dropout=0.1):
        super().__init__()

        # 动作编码器
        self.action_encoder = ActionEncoder(action_dim=action_dim, output_dim=d_model)

        # 目标-动作融合层
        self.goal_action_fusion = nn.Sequential(
            nn.Linear(d_model + goal_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Transformer层
        self.layers = nn.ModuleList([
            HierarchicalTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 输出层归一化
        self.output_ln = nn.LayerNorm(d_model)

    def forward(self, state_features, actions, goal_features):
        """
        Args:
            state_features: 编码器输出 (B, d_model)
            actions: 动作索引 (B,)
            goal_features: 目标特征 (B, goal_dim)
        Returns:
            predicted_features: 预测特征 (B, d_model)
        """
        # 编码动作
        action_features = self.action_encoder(actions)

        # 融合目标和动作信息
        goal_action_combined = torch.cat([action_features, goal_features], dim=-1)
        goal_action_features = self.goal_action_fusion(goal_action_combined)

        # 通过Transformer层
        x = state_features
        for layer in self.layers:
            x = layer(x, goal_action_features)

        # 输出归一化
        x = self.output_ln(x)

        return x


# ============================================================================
# 主模型：分层PRM_JE
# ============================================================================

class HierarchicalPRM_JE(nn.Module):
    """分层预测表示模型 - 集成元控制器和控制器"""

    def __init__(
            self,
            img_in_channels: int = 3,
            encoder_layers: List[int] = [2, 3, 4, 3],
            action_dim: int = 19,
            latent_dim: int = 256,
            base_channels: int = 64,
            num_attention_heads: int = 8,
            transformer_layers: int = 3,
            loss_weight: float = 1.0,
            dropout: float = 0.1,
            # 新增：分层参数
            goal_dim: int = 64,
            num_goal_types: int = 3,
            meta_hidden_dim: int = 512
    ) -> None:
        super().__init__()

        # 原有编码器
        self.encoder = Encoder(img_in_channels, encoder_layers, base_channels)

        # === 新增：分层组件 ===
        # 1. 目标编码器
        self.goal_encoder = GoalEncoder(goal_dim, latent_dim)

        # 2. 元控制器
        self.meta_controller = MetaController(latent_dim, meta_hidden_dim, num_goal_types)

        # 3. 分层预测模型（替换原有的predictive_model）
        self.predictive_model = HierarchicalPredictiveModel(
            d_model=latent_dim,
            goal_dim=latent_dim,  # 目标编码后的维度
            action_dim=action_dim,
            n_heads=num_attention_heads,
            n_layers=transformer_layers,
            dropout=dropout
        )

        # 4. 内在奖励计算器
        self.intrinsic_critic = IntrinsicCritic(latent_dim, latent_dim)

        # 其他参数
        self.loss_weight = loss_weight
        self.goal_types = ['spatial', 'object', 'exploration']

        # 分层状态跟踪
        self.current_goal_type = None
        self.current_goal_params = None
        self.goal_step_count = 0
        self.goal_selection_interval = 5

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """参考LLM的权重初始化策略"""
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

    def select_goal(self, state_features: torch.Tensor, epsilon: float = 0.1) -> Tuple[str, torch.Tensor]:
        """
        选择目标（epsilon-greedy策略）
        Args:
            state_features: 状态特征 (B, latent_dim)
            epsilon: 探索概率
        Returns:
            goal_type: 选择的目标类型
            goal_params: 目标参数
        """
        if np.random.random() < epsilon:
            # 随机选择目标
            goal_type = np.random.choice(self.goal_types)
        else:
            # 基于Q值选择目标
            with torch.no_grad():
                goal_q_values, _ = self.meta_controller(state_features)
                goal_idx = torch.argmax(goal_q_values, dim=1)[0].item()
                goal_type = self.goal_types[goal_idx]

        # 生成目标参数
        with torch.no_grad():
            _, goal_params = self.meta_controller(state_features, goal_type)

        return goal_type, goal_params

    def forward(self, images: torch.Tensor, actions: torch.Tensor,
                goal_type: str = None, goal_params: torch.Tensor = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            images: 输入图像 (B, 3, 210, 160)
            actions: 动作 (B,) - 0到18的整数
            goal_type: 目标类型
            goal_params: 目标参数
        Returns:
            encoder_features: 编码器输出 (B, 256)
            predicted_features: 预测模型输出 (B, 256)
            goal_features: 目标特征 (B, 256)
        """
        # 确保动作在正确范围内
        assert actions.min() >= 0 and actions.max() <= 18, f"Actions must be in range [0, 18]"

        # 编码
        encoder_features = self.encoder(images)

        # 目标处理
        if goal_type is None or goal_params is None:
            # 自动选择目标
            goal_type, goal_params = self.select_goal(encoder_features)

        # 编码目标
        goal_features = self.goal_encoder(goal_type, goal_params)

        # 分层预测
        predicted_features = self.predictive_model(encoder_features, actions, goal_features)

        return encoder_features, predicted_features, goal_features

    def compute_controller_loss(self, predicted_features: torch.Tensor, target_features: torch.Tensor,
                                current_features: torch.Tensor, goal_features: torch.Tensor) -> Tuple[
        torch.Tensor, dict]:
        """
        计算控制器损失（预测损失 + 内在奖励损失）
        """
        # 1. 预测损失（原有逻辑）
        pred_norm = F.normalize(predicted_features, p=2, dim=1)
        target_norm = F.normalize(target_features, p=2, dim=1)
        cosine_sim = (pred_norm * target_norm).sum(dim=1)
        prediction_loss = (1.0 - cosine_sim).mean()

        # 2. 内在奖励损失
        intrinsic_reward = self.intrinsic_critic(target_features, goal_features)
        # 鼓励高内在奖励（目标导向）
        intrinsic_loss = F.mse_loss(intrinsic_reward, torch.ones_like(intrinsic_reward))

        # 总控制器损失
        total_loss = self.loss_weight * prediction_loss + 0.1 * intrinsic_loss

        loss_dict = {
            'controller_total_loss': total_loss.item(),
            'prediction_loss': prediction_loss.item(),
            'intrinsic_loss': intrinsic_loss.item(),
            'cosine_similarity': cosine_sim.mean().item(),
            'intrinsic_reward': intrinsic_reward.mean().item()
        }

        return total_loss, loss_dict

    def compute_meta_loss(self, transitions: List[Dict]) -> Tuple[torch.Tensor, dict]:
        """
        计算元控制器损失（基于累积外在奖励的Q-learning）
        """
        if not transitions:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device), {}

        total_loss = 0
        count = 0

        for trans in transitions:
            state_features = trans['state_features']
            goal_type_idx = self.goal_types.index(trans['goal_type'])
            extrinsic_reward = trans['extrinsic_reward']
            next_state_features = trans['next_state_features']
            done = trans['done']

            # 当前Q值
            current_q_values, _ = self.meta_controller(state_features)
            current_q = current_q_values[0, goal_type_idx]

            # 目标Q值
            with torch.no_grad():
                if done:
                    target_q = extrinsic_reward
                else:
                    next_q_values, _ = self.meta_controller(next_state_features)
                    target_q = extrinsic_reward + 0.99 * torch.max(next_q_values)

            loss = F.mse_loss(current_q, target_q)
            total_loss += loss
            count += 1

        avg_loss = total_loss / count if count > 0 else total_loss

        loss_dict = {
            'meta_loss': avg_loss.item(),
            'num_transitions': count
        }

        return avg_loss, loss_dict


# ============================================================================
# 模型分析工具（现在可以安全引用HierarchicalPRM_JE）
# ============================================================================

class HierarchicalModelAnalyzer:
    """分层模型分析工具"""

    def __init__(self, model: HierarchicalPRM_JE):
        self.model = model

    def analyze_goal_distribution(self, images: torch.Tensor, num_samples: int = 100):
        """分析目标分布"""
        goal_counts = {goal_type: 0 for goal_type in self.model.goal_types}

        with torch.no_grad():
            # 先编码图像为特征
            state_features = self.model.encoder(images[:1])

            for _ in range(num_samples):
                goal_type, _ = self.model.select_goal(state_features, epsilon=0.0)
                goal_counts[goal_type] += 1

        return {k: v / num_samples for k, v in goal_counts.items()}

    def get_goal_q_values(self, images: torch.Tensor):
        """获取目标Q值"""
        with torch.no_grad():
            # 编码图像为特征
            features = self.model.encoder(images)
            q_values, _ = self.model.meta_controller(features)
            return q_values

    def compute_intrinsic_reward_statistics(self, images: torch.Tensor, goal_type: str):
        """计算内在奖励统计"""
        with torch.no_grad():
            # 编码图像为特征
            features = self.model.encoder(images)
            _, goal_params = self.model.meta_controller(features[:1], goal_type)
            goal_features = self.model.goal_encoder(goal_type, goal_params.expand(len(features), -1))
            intrinsic_rewards = self.model.intrinsic_critic(features, goal_features)

            return {
                'mean': intrinsic_rewards.mean().item(),
                'std': intrinsic_rewards.std().item(),
                'min': intrinsic_rewards.min().item(),
                'max': intrinsic_rewards.max().item()
            }


# ============================================================================
# 辅助函数
# ============================================================================

def create_hierarchical_optimizers(model, controller_lr=1e-4, meta_lr=1e-5, weight_decay=1e-2):
    """创建分层优化器"""

    # 控制器参数（编码器 + 预测模型 + 内在奖励）
    controller_params = list(model.encoder.parameters()) + \
                        list(model.predictive_model.parameters()) + \
                        list(model.intrinsic_critic.parameters()) + \
                        list(model.goal_encoder.parameters())

    # 元控制器参数
    meta_params = list(model.meta_controller.parameters())

    # 分别创建优化器
    controller_optimizer = torch.optim.AdamW(
        controller_params,
        lr=controller_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay
    )

    meta_optimizer = torch.optim.AdamW(
        meta_params,
        lr=meta_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay
    )

    return controller_optimizer, meta_optimizer


def create_hierarchical_lr_schedulers(controller_optimizer, meta_optimizer,
                                      controller_warmup_steps=1000, meta_warmup_steps=500,
                                      controller_max_steps=10000, meta_max_steps=5000):
    """创建分层学习率调度器"""

    def controller_lr_lambda(step):
        if step < controller_warmup_steps:
            return step / controller_warmup_steps
        else:
            progress = (step - controller_warmup_steps) / (controller_max_steps - controller_warmup_steps)
            progress = min(progress, 1.0)
            return 0.1 + 0.9 * (0.5 * (1 + math.cos(math.pi * progress)))

    def meta_lr_lambda(step):
        if step < meta_warmup_steps:
            return step / meta_warmup_steps
        else:
            progress = (step - meta_warmup_steps) / (meta_max_steps - meta_warmup_steps)
            progress = min(progress, 1.0)
            return 0.1 + 0.9 * (0.5 * (1 + math.cos(math.pi * progress)))

    controller_scheduler = torch.optim.lr_scheduler.LambdaLR(controller_optimizer, controller_lr_lambda)
    meta_scheduler = torch.optim.lr_scheduler.LambdaLR(meta_optimizer, meta_lr_lambda)

    return controller_scheduler, meta_scheduler


# ============================================================================
# 兼容性函数 - 保持与原有代码的兼容性
# ============================================================================

def create_optimized_optimizer(model, base_lr=1e-4, weight_decay=1e-2):
    """
    兼容原有代码的优化器创建函数
    对于分层模型，这个函数只创建控制器优化器
    """
    if isinstance(model, HierarchicalPRM_JE):
        # 如果是分层模型，只返回控制器优化器
        controller_optimizer, _ = create_hierarchical_optimizers(
            model, controller_lr=base_lr, meta_lr=base_lr * 0.1, weight_decay=weight_decay
        )
        return controller_optimizer
    else:
        # 原有的优化器逻辑（保持兼容性）
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) <= 1 or name.endswith(".bias") or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": weight_decay,
                "lr": base_lr
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
                "lr": base_lr
            }
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=base_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay
        )

        return optimizer


def create_lr_scheduler(optimizer, warmup_steps=1000, max_steps=10000, min_lr_ratio=0.1):
    """
    兼容原有代码的学习率调度器创建函数
    """

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


# ============================================================================
# 工厂函数 - 根据配置创建合适的模型
# ============================================================================

def create_prm_model(model_type='hierarchical', **kwargs):
    """
    工厂函数：根据类型创建合适的PRM模型
    Args:
        model_type: 模型类型 ('hierarchical' 或 'original')
        **kwargs: 模型参数
    """
    if model_type == 'hierarchical':
        return HierarchicalPRM_JE(**kwargs)
    else:
        # 这里可以添加原有PRM_JE的导入和创建逻辑
        raise NotImplementedError("Original PRM_JE not implemented in this file")


# ============================================================================
# 测试和演示代码
# ============================================================================

if __name__ == '__main__':
    # === 测试分层模型 ===
    print("=== Hierarchical PRM_JE Model Testing ===")

    BATCH_SIZE = 4
    IMG_C = 3
    IMG_H = 210
    IMG_W = 160
    ACTION_DIM = 19

    # 创建测试数据
    dummy_images = torch.rand(BATCH_SIZE, IMG_C, IMG_H, IMG_W)
    dummy_actions = torch.randint(0, ACTION_DIM, (BATCH_SIZE,))
    dummy_next_images = torch.rand(BATCH_SIZE, IMG_C, IMG_H, IMG_W)

    # 创建分层模型
    model = HierarchicalPRM_JE(
        img_in_channels=IMG_C,
        encoder_layers=[2, 3, 4, 3],
        action_dim=ACTION_DIM,
        latent_dim=256,
        base_channels=64,
        num_attention_heads=8,
        transformer_layers=3,
        loss_weight=1.0,
        dropout=0.1,
        goal_dim=64,
        num_goal_types=3,
        meta_hidden_dim=512
    )

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    controller_params = sum(p.numel() for p in
                            list(model.encoder.parameters()) +
                            list(model.predictive_model.parameters()) +
                            list(model.intrinsic_critic.parameters()) +
                            list(model.goal_encoder.parameters()))
    meta_params = sum(p.numel() for p in model.meta_controller.parameters())

    print(f"总参数数量: {total_params:,}")
    print(f"控制器参数: {controller_params:,}")
    print(f"元控制器参数: {meta_params:,}")

    # 测试前向传播
    print(f"\n=== 前向传播测试 ===")
    model.eval()
    with torch.no_grad():
        encoder_features, predicted_features, goal_features = model(dummy_images, dummy_actions)
        print(f"✅ 编码器特征: {encoder_features.shape}")
        print(f"✅ 预测特征: {predicted_features.shape}")
        print(f"✅ 目标特征: {goal_features.shape}")

    # 创建分层优化器
    controller_optimizer, meta_optimizer = create_hierarchical_optimizers(model)
    controller_scheduler, meta_scheduler = create_hierarchical_lr_schedulers(
        controller_optimizer, meta_optimizer
    )

    print(f"\n=== 分层训练测试 ===")
    n_epochs = 20

    for epoch in range(n_epochs):
        model.train()

        # === 控制器训练步骤 ===
        controller_optimizer.zero_grad()

        # 前向传播
        encoder_features, predicted_features, goal_features = model(dummy_images, dummy_actions)

        # 目标特征
        with torch.no_grad():
            target_features = model.encoder(dummy_next_images)

        # 计算控制器损失
        controller_loss, controller_loss_dict = model.compute_controller_loss(
            predicted_features, target_features, encoder_features, goal_features
        )

        # 控制器反向传播
        controller_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.encoder.parameters()) +
            list(model.predictive_model.parameters()) +
            list(model.intrinsic_critic.parameters()) +
            list(model.goal_encoder.parameters()),
            max_norm=1.0
        )
        controller_optimizer.step()
        controller_scheduler.step()

        # === 元控制器训练步骤（分离计算图）===
        if epoch % 5 == 0:  # 每5步更新元控制器
            meta_optimizer.zero_grad()

            # 重新编码特征（避免计算图冲突）
            with torch.no_grad():
                meta_state_features = model.encoder(dummy_images[0:1])
                meta_next_features = model.encoder(dummy_next_images[0:1])

            meta_transitions = [{
                'state_features': meta_state_features,
                'goal_type': 'spatial',
                'extrinsic_reward': torch.tensor(0.1),
                'next_state_features': meta_next_features,
                'done': False
            }]

            meta_loss, meta_loss_dict = model.compute_meta_loss(meta_transitions)

            if meta_loss.item() > 0:
                meta_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.meta_controller.parameters(), max_norm=1.0)
                meta_optimizer.step()
                meta_scheduler.step()

        if (epoch + 1) % 5 == 0:
            controller_lr = controller_optimizer.param_groups[0]['lr']
            meta_lr = meta_optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch + 1:3d}/{n_epochs}, '
                  f'Controller Loss: {controller_loss_dict["controller_total_loss"]:.6f}, '
                  f'Prediction: {controller_loss_dict["prediction_loss"]:.6f}, '
                  f'Intrinsic: {controller_loss_dict["intrinsic_loss"]:.6f}, '
                  f'C_LR: {controller_lr:.2e}, M_LR: {meta_lr:.2e}')

    # 测试模型分析器
    print(f"\n=== 模型分析测试 ===")
    analyzer = HierarchicalModelAnalyzer(model)

    # 分析目标分布（使用较少样本以加快测试）
    goal_distribution = analyzer.analyze_goal_distribution(dummy_images, num_samples=50)
    print(f"目标分布: {goal_distribution}")

    # 获取Q值
    q_values = analyzer.get_goal_q_values(dummy_images)
    print(f"目标Q值形状: {q_values.shape}")
    print(f"目标Q值: {q_values[0].tolist()}")

    # 计算内在奖励统计
    for goal_type in model.goal_types:
        stats = analyzer.compute_intrinsic_reward_statistics(dummy_images, goal_type)
        print(f"{goal_type}目标内在奖励统计: {stats}")

    print(f"\n=== 分层PRM_JE模型总结 ===")
    print(f"✅ 编码器: 处理210×160×3图像 → 256维特征")
    print(f"✅ 目标编码器: 支持空间、对象、探索三种目标类型")
    print(f"✅ 元控制器: 基于状态选择目标类型和参数")
    print(f"✅ 分层预测: 结合目标信息的Transformer预测")
    print(f"✅ 内在奖励: 评估目标完成情况")
    print(f"✅ 双层优化: 控制器和元控制器分别优化")
    print(f"✅ 模型分析: 支持目标分布和奖励统计分析")
    print(f"✅ 总参数: {total_params:,}")
    print("🎉 分层PRM_JE模型测试完成！")
