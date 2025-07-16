# enhanced_ppm_attention.py
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import nn
import os
import math
from typing import Optional

# 假设 .residual 包含 DownBlock2d 和 UpBlock2d。
# 为了独立测试，如果不可用，我们定义增强版本。
try:
    from .residual import DownBlock2d, UpBlock2d
except ImportError:

    # --- SE注意力模块 ---
    # 让神经网络“学会关注哪些通道更重要;对每个通道进行加权，增强有用特征、抑制无效特征。
    class SEBlock(nn.Module):
        """Squeeze-and-Excitation BlockSE"""
        """SE注意力机制通过学习通道间的重要性权重来增强特征表达"""

        def __init__(self, channels, reduction=16):
            super().__init__()
            self.squeeze = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
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


    # --- 增强残差块 ---
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

            # 主卷积层
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

            # 残差块
            self.residual_blocks = nn.ModuleList([
                EnhancedResidualBlock(out_channels, 2, use_se) for _ in range(num_layers)
            ])

            # shortcut连接
            self.shortcut = nn.Identity()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            identity = self.shortcut(x)

            # 主路径
            out = self.conv(x)
            out = self.bn(out)
            out = self.relu(out)

            '''尺寸自适应'''
            if self.downsample and identity.shape[2:] != out.shape[2:]:
                identity = F.adaptive_avg_pool2d(identity, out.shape[2:])

            out = out + identity

            for res_block in self.residual_blocks:
                out = res_block(out)

            return out


    class UpBlock2d(nn.Module):
        """增强的上采样块，支持跳跃连接"""

        def __init__(self, in_channels, out_channels, num_layers, upsample=False, skip_channels=0):
            super().__init__()
            self.upsample_layer = None
            if upsample:
                # 通过权重进行插值、学习上采样方式
                self.upsample_layer = nn.ConvTranspose2d(
                    in_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False
                )
                self.upsample_bn = nn.BatchNorm2d(in_channels)
                self.upsample_relu = nn.ReLU(inplace=True)

            # 如果有跳跃连接，调整输入通道数
            conv_in_channels = in_channels + skip_channels

            self.conv = nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

            # 残差块
            self.residual_blocks = nn.ModuleList([
                EnhancedResidualBlock(out_channels, 2) for _ in range(num_layers)
            ])

        def forward(self, x, skip=None):
            if self.upsample_layer:
                x = self.upsample_layer(x)
                x = self.upsample_bn(x)
                x = self.upsample_relu(x)

            # 如果有跳跃连接，进行特征融合
            if skip is not None:
                # 确保尺寸匹配
                if x.shape[2:] != skip.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)

            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)

            # 通过残差块
            for res_block in self.residual_blocks:
                x = res_block(x)

            return x


# --- 多头动作注意力机制 ---
class MultiHeadActionAttention(nn.Module):
    """多头动作注意力机制"""

    def __init__(self, action_dim=18, latent_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        assert latent_dim % num_heads == 0

        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # 动作嵌入层
        self.action_embedding = nn.Sequential(
            nn.Embedding(action_dim, 128),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim)
        )

        # 多头注意力投影层
        self.q_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.k_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.v_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

        # 位置编码？---是否需要，编码的是什么的位置？是否可以用动作序列的内容
        self.pos_encoding = nn.Parameter(torch.randn(1, latent_dim) * 0.02)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.ln1 = nn.LayerNorm(latent_dim)
        self.ln2 = nn.LayerNorm(latent_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim)
        )

    def forward(self, encoded_features, actions):
        """
        Args:
            encoded_features: 编码器输出 (B, latent_dim)
            actions: 动作索引 (B,)
        Returns:
            attended_features: 注意力加权后的特征 (B, latent_dim)
        """
        B = encoded_features.size(0)

        # 动作嵌入
        action_features = self.action_embedding(actions.long())  # (B, latent_dim)

        # 添加位置编码
        visual_features = encoded_features + self.pos_encoding  # (B, latent_dim)

        # Layer normalization
        visual_norm = self.ln1(visual_features)
        action_norm = self.ln1(action_features)

        # 多头注意力计算
        Q = self.q_proj(action_norm).view(B, self.num_heads, self.head_dim)  # (B, H, D)
        K = self.k_proj(visual_norm).view(B, self.num_heads, self.head_dim)  # (B, H, D)
        V = self.v_proj(visual_norm).view(B, self.num_heads, self.head_dim)  # (B, H, D)

        # 注意力分数计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, 1)
        attn_weights = F.softmax(scores, dim=1)  # 在头维度上softmax
        attn_weights = self.dropout(attn_weights)

        # 应用注意力
        attended = torch.matmul(attn_weights, V)  # (B, H, D)
        attended = attended.view(B, self.latent_dim)  # (B, latent_dim)

        # 输出投影
        attended = self.out_proj(attended)

        # 残差连接
        output = visual_features + attended

        # Feed-forward network with residual connection
        output = output + self.ffn(self.ln2(output))

        return output


# --- 增强编码器 ---
class EnhancedEncoder(nn.Module):
    """增强的编码器，更深的网络结构"""

    def __init__(self, in_channels: int, layers: List[int], base_channels: int = 64) -> None:
        super().__init__()

        # Stem层
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels, kernel_size=3, stride=2, padding=1, bias=False),  # H/2
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # 更深的特征提取网络
        self.layer1 = DownBlock2d(base_channels, base_channels, layers[0], downsample=False)
        self.layer2 = DownBlock2d(base_channels, base_channels * 2, layers[1], downsample=True)  # H/4
        self.layer3 = DownBlock2d(base_channels * 2, base_channels * 4, layers[2], downsample=True)  # H/8
        self.layer4 = DownBlock2d(base_channels * 4, base_channels * 8, layers[3], downsample=True)  # H/16

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 最终的全连接层
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )

        # 存储中间特征用于跳跃连接
        self.skip_features = []

    def forward(self, x: torch.Tensor) -> tuple:
        self.skip_features = []

        x = self.stem(x)  # (B, 64, H/2, W/2)
        self.skip_features.append(x)

        x = self.layer1(x)  # (B, 64, H/2, W/2)
        self.skip_features.append(x)

        x = self.layer2(x)  # (B, 128, H/4, W/4)
        self.skip_features.append(x)

        x = self.layer3(x)  # (B, 256, H/8, W/8)
        self.skip_features.append(x)

        x = self.layer4(x)  # (B, 512, H/16, W/16)

        # 全局池化和特征向量生成
        pooled = self.global_pool(x)  # (B, 512, 1, 1)
        flattened = pooled.view(pooled.size(0), -1)  # (B, 512)
        features = self.fc(flattened)  # (B, 256)

        return features, self.skip_features


# --- 增强解码器 ---
class EnhancedDecoder(nn.Module):
    """增强的解码器，支持跳跃连接"""

    def __init__(self, latent_dim: int, out_channels: int, layers: List[int],
                 base_channels: int = 64, use_skip_connections: bool = True) -> None:
        super().__init__()
        self.use_skip_connections = use_skip_connections

        # 初始投影层
        self.initial_size = 8  # 假设最小特征图为8x8
        self.initial_channels = base_channels * 8  # 512
        self.initial_proj = nn.Sequential(
            nn.Linear(latent_dim, self.initial_channels * self.initial_size * self.initial_size),
            nn.ReLU(inplace=True)
        )

        # 上采样块 - 支持跳跃连接
        skip_channels = [256, 128, 64, 64] if use_skip_connections else [0, 0, 0, 0]

        self.up1 =UpBlock2d(
            self.initial_channels, base_channels * 4, layers[0],
            upsample=True, skip_channels=skip_channels[0]
        )  # 16x16
        self.up2 = UpBlock2d(
            base_channels * 4, base_channels * 2, layers[1],
            upsample=True, skip_channels=skip_channels[1]
        )  # 32x32
        self.up3 = UpBlock2d(
            base_channels * 2, base_channels, layers[2],
            upsample=True, skip_channels=skip_channels[2]
        )  # 64x64
        self.up4 = UpBlock2d(
            base_channels, base_channels, layers[3] if len(layers) > 3 else 1,
            upsample=True, skip_channels=skip_channels[3]
        )  # 128x128

        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # 输出范围[-1,1]
        )

    def forward(self, x: torch.Tensor, skip_features: Optional[List] = None) -> torch.Tensor:
        # 投影到初始特征图
        x = self.initial_proj(x)
        x = x.view(x.size(0), self.initial_channels, self.initial_size, self.initial_size)

        # 渐进式上采样
        if self.use_skip_connections and skip_features is not None:
            # 反向使用跳跃连接
            skip_list = skip_features[::-1]  # 倒序
            x = self.up1(x, skip_list[0] if len(skip_list) > 0 else None)
            x = self.up2(x, skip_list[1] if len(skip_list) > 1 else None)
            x = self.up3(x, skip_list[2] if len(skip_list) > 2 else None)
            x = self.up4(x, skip_list[3] if len(skip_list) > 3 else None)
        else:
            x = self.up1(x)
            x = self.up2(x)
            x = self.up3(x)
            x = self.up4(x)

        x = self.final_conv(x)
        return x


# --- 增强预测编码器模型 ---
class EnhancedPredictiveRepModel(nn.Module):
    def __init__(
            self, img_in_channels: int, img_out_channels: int,
            encoder_layers: List[int], decoder_layers: List[int],
            target_img_h: int, target_img_w: int,
            action_dim: int = 18, latent_dim: int = 256,
            base_channels: int = 64, num_attention_heads: int = 8,
            use_skip_connections: bool = True
    ) -> None:
        super().__init__()

        # 增强的编码器
        self.encoder = EnhancedEncoder(img_in_channels, encoder_layers, base_channels)

        # 多头动作注意力机制
        self.attention = MultiHeadActionAttention(
            action_dim=action_dim,
            latent_dim=latent_dim,
            num_heads=num_attention_heads
        )

        # 增强的解码器
        self.decoder = EnhancedDecoder(
            latent_dim, img_out_channels, decoder_layers,
            base_channels, use_skip_connections
        )

        self.target_img_h = target_img_h
        self.target_img_w = target_img_w
        self.use_skip_connections = use_skip_connections

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, x_obs1: torch.Tensor, actions: torch.Tensor):
        """
        前向传播。
        参数:
            x_obs1 (torch.Tensor): 输入观测 obs1，形状 (B, L, C_in, H_in, W_in)。L 通常为 1。
            actions (torch.Tensor): 动作张量，形状 (B, L)。L 通常为 1。
        返回:
            o_latent (torch.Tensor): 编码器输出的潜向量，形状 (B, L, 256)。
            output_x_reconstructed (torch.Tensor): 解码器重建的下一帧观测。
        """
        B, L, C_in, H_in, W_in = x_obs1.shape
        # 将输入 reshape 以便编码器处理: (B*L, C_in, H_in, W_in)
        x_reshaped = x_obs1.reshape(B * L, C_in, H_in, W_in)
        actions_reshaped = actions.reshape(B * L)

        # 1. 编码图像到 256 维向量，同时获取跳跃连接特征
        encoded_vector, skip_features = self.encoder(x_reshaped)  # (B*L, 256), [skip_features]

        # 2. 使用多头注意力机制处理编码特征和动作
        attended_features = self.attention(encoded_vector, actions_reshaped)  # (B*L, 256)

        # 3. 解码
        if self.use_skip_connections:
            decoded_frames_raw = self.decoder(attended_features, skip_features)
        else:
            decoded_frames_raw = self.decoder(attended_features)

        # 将输出 reshape 回原始批次格式
        final_out_C, final_out_H, final_out_W = decoded_frames_raw.shape[1:]
        output_x_reconstructed = decoded_frames_raw.reshape(B, L, final_out_C, final_out_H, final_out_W)
        o_latent_return = encoded_vector.reshape(B, L, 256)

        return o_latent_return, output_x_reconstructed

    def _crop_output(self, predicted_obs_raw: torch.Tensor) -> torch.Tensor:
        """裁剪模型输出以匹配目标图像尺寸"""
        current_h, current_w = predicted_obs_raw.shape[2], predicted_obs_raw.shape[3]
        h_slice = slice(0, min(current_h, self.target_img_h))
        w_slice = slice(0, min(current_w, self.target_img_w))

        cropped_obs = predicted_obs_raw[:, :, h_slice, w_slice]
        return cropped_obs

    def train_on_batch(self, obs1_input_b_l_c_h_w: torch.Tensor,
                       actions_input_b_l: torch.Tensor,
                       obs2_target_b_l_c_h_w: torch.Tensor,
                       optimizer: torch.optim.Optimizer) -> float:
        """在单个批次上训练模型 (修改后，与 evaluate_on_batch 损失计算保持一致)"""
        self.train()
        optimizer.zero_grad()

        # 前向传播
        _, predicted_obs2_raw_b_l_c_hd_wd = self.forward(obs1_input_b_l_c_h_w, actions_input_b_l)

        # 处理预测输出和目标
        predicted_obs2_processed_b_c_hd_wd = predicted_obs2_raw_b_l_c_hd_wd.squeeze(1)
        obs2_target_for_loss_b_c_h_w = obs2_target_b_l_c_h_w.squeeze(1)

        # 尺寸匹配处理
        if predicted_obs2_processed_b_c_hd_wd.shape[2:] != obs2_target_for_loss_b_c_h_w.shape[2:]:
            predicted_obs2_resized = F.interpolate(
                predicted_obs2_processed_b_c_hd_wd,
                size=obs2_target_for_loss_b_c_h_w.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        else:
            predicted_obs2_resized = predicted_obs2_processed_b_c_hd_wd

        # 将预测输出和目标从 [-1, 1] 映射到 [0, 1]
        predicted_norm = (predicted_obs2_resized + 1) / 2.0
        target_norm = (obs2_target_for_loss_b_c_h_w + 1) / 2.0
        
        print("predicted_obs2_resized: min =", predicted_obs2_resized.min().item(), ", max =", predicted_obs2_resized.max().item())
        print("predicted_norm: min =", predicted_norm.min().item(), ", max =", predicted_norm.max().item())
        print("obs2_target_for_loss_b_c_h_w: min =", obs2_target_for_loss_b_c_h_w.min().item(), ", max =", obs2_target_for_loss_b_c_h_w.max().item())
        print("target_norm: min =", target_norm.min().item(), ", max =", target_norm.max().item())
        
        # 转换为灰度图像
        gray_weights = torch.tensor([0.2989, 0.5870, 0.1140],
                                      device=predicted_norm.device).view(1, -1, 1, 1)
        predicted_gray = (predicted_norm * gray_weights).sum(dim=1, keepdim=True)
        target_gray = (target_norm * gray_weights).sum(dim=1, keepdim=True)

        # 计算 MSE 损失
        loss = F.mse_loss(predicted_gray, target_gray)

        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        optimizer.step()

        return loss.item()

    def train_on_batch_advanced(self, obs1_input_b_l_c_h_w: torch.Tensor,
                           actions_input_b_l: torch.Tensor,
                           obs2_target_b_l_c_h_w: torch.Tensor,
                           optimizer: torch.optim.Optimizer,
                           scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                           scaler: Optional[torch.cuda.amp.GradScaler] = None,
                           use_mixed_precision: bool = False,
                           gradient_clip_norm: float = 1.0,
                           accumulation_steps: int = 1,
                           is_accumulation_step: bool = False) -> float:
        # 单个批次上训练模型，支持梯度累积和混合精度训练
        self.train()
        # Only zero gradients at the beginning of accumulation cycle
        if not is_accumulation_step:
            optimizer.zero_grad()

        if use_mixed_precision and scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                # Forward pass
                _, predicted_obs2_raw_b_l_c_hd_wd = self.forward(obs1_input_b_l_c_h_w, actions_input_b_l)

                # Process predictions and targets
                predicted_obs2_processed_b_c_hd_wd = predicted_obs2_raw_b_l_c_hd_wd.squeeze(1)
                obs2_target_for_loss_b_c_h_w = obs2_target_b_l_c_h_w.squeeze(1)

                # Size matching
                if predicted_obs2_processed_b_c_hd_wd.shape[2:] != obs2_target_for_loss_b_c_h_w.shape[2:]:
                    predicted_obs2_resized = F.interpolate(
                        predicted_obs2_processed_b_c_hd_wd,
                        size=obs2_target_for_loss_b_c_h_w.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    predicted_obs2_resized = predicted_obs2_processed_b_c_hd_wd

                # Normalize to [0, 1] range
                predicted_norm = (predicted_obs2_resized + 1) / 2.0
                target_norm = (obs2_target_for_loss_b_c_h_w + 1) / 2.0

                # Convert to grayscale
                gray_weights = torch.tensor([0.2989, 0.5870, 0.1140],
                                            device=predicted_norm.device).view(1, -1, 1, 1)
                predicted_gray = (predicted_norm * gray_weights).sum(dim=1, keepdim=True)
                target_gray = (target_norm * gray_weights).sum(dim=1, keepdim=True)

                # Compute loss and scale for accumulation
                loss = F.mse_loss(predicted_gray, target_gray)
                loss = loss / accumulation_steps

            # Backward pass with scaling
            scaler.scale(loss).backward()

            # Only step optimizer at the end of accumulation cycle
            if not is_accumulation_step:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=gradient_clip_norm)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                
                # Update learning rate
                if scheduler is not None:
                    scheduler.step()

        else:
            # Standard precision training
            # Forward pass
            _, predicted_obs2_raw_b_l_c_hd_wd = self.forward(obs1_input_b_l_c_h_w, actions_input_b_l)

            # Process predictions and targets (same as above)
            predicted_obs2_processed_b_c_hd_wd = predicted_obs2_raw_b_l_c_hd_wd.squeeze(1)
            obs2_target_for_loss_b_c_h_w = obs2_target_b_l_c_h_w.squeeze(1)

            # Size matching
            if predicted_obs2_processed_b_c_hd_wd.shape[2:] != obs2_target_for_loss_b_c_h_w.shape[2:]:
                predicted_obs2_resized = F.interpolate(
                    predicted_obs2_processed_b_c_hd_wd,
                    size=obs2_target_for_loss_b_c_h_w.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                predicted_obs2_resized = predicted_obs2_processed_b_c_hd_wd

            # Normalize to [0, 1] range
            predicted_norm = (predicted_obs2_resized + 1) / 2.0
            target_norm = (obs2_target_for_loss_b_c_h_w + 1) / 2.0

            # Convert to grayscale
            gray_weights = torch.tensor([0.2989, 0.5870, 0.1140],
                                        device=predicted_norm.device).view(1, -1, 1, 1)
            predicted_gray = (predicted_norm * gray_weights).sum(dim=1, keepdim=True)
            target_gray = (target_norm * gray_weights).sum(dim=1, keepdim=True)

            # Compute loss and scale for accumulation
            loss = F.mse_loss(predicted_gray, target_gray)
            loss = loss / accumulation_steps

            # Backward pass
            loss.backward()

            # Only step optimizer at the end of accumulation cycle
            if not is_accumulation_step:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=gradient_clip_norm)
                
                # Optimizer step
                optimizer.step()
                
                # Update learning rate
                if scheduler is not None:
                    scheduler.step()

        return loss.item() * accumulation_steps  # Return unscaled loss for logging

    def evaluate_on_batch(self, obs1_input_b_l_c_h_w: torch.Tensor,
                          actions_input_b_l: torch.Tensor,
                          obs2_target_b_l_c_h_w: torch.Tensor) -> float:
        """在单个批次上评估模型"""
        self.eval()
        with torch.no_grad():
            _, predicted_obs2_raw_b_l_c_hd_wd = self.forward(obs1_input_b_l_c_h_w, actions_input_b_l)

            predicted_obs2_processed_b_c_hd_wd = predicted_obs2_raw_b_l_c_hd_wd.squeeze(1)
            obs2_target_for_loss_b_c_h_w = obs2_target_b_l_c_h_w.squeeze(1)

            # 尺寸匹配处理
            if predicted_obs2_processed_b_c_hd_wd.shape[2:] != obs2_target_for_loss_b_c_h_w.shape[2:]:
                predicted_obs2_resized = F.interpolate(
                    predicted_obs2_processed_b_c_hd_wd,
                    size=obs2_target_for_loss_b_c_h_w.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                predicted_obs2_resized = predicted_obs2_processed_b_c_hd_wd

            # 将预测输出和目标从 [-1, 1] 映射到 [0, 1]
            predicted_norm = (predicted_obs2_resized + 1) / 2.0
            target_norm = (obs2_target_for_loss_b_c_h_w + 1) / 2.0

            # Debug 输出
            # print("predicted_obs2_resized: min =", predicted_obs2_resized.min().item(), ", max =", predicted_obs2_resized.max().item())
            # print("predicted_norm: min =", predicted_norm.min().item(), ", max =", predicted_norm.max().item())
            # print("obs2_target_for_loss_b_c_h_w: min =", obs2_target_for_loss_b_c_h_w.min().item(), ", max =", obs2_target_for_loss_b_c_h_w.max().item())
            # print("target_norm: min =", target_norm.min().item(), ", max =", target_norm.max().item())

            # 转换为灰度图像
            gray_weights = torch.tensor([0.2989, 0.5870, 0.1140],
                                        device=predicted_norm.device).view(1, -1, 1, 1)
            predicted_gray = (predicted_norm * gray_weights).sum(dim=1, keepdim=True)
            target_gray = (target_norm * gray_weights).sum(dim=1, keepdim=True)

            # 计算 MSE 损失
            loss = F.mse_loss(predicted_gray, target_gray)

        return loss.item()


if __name__ == '__main__':
    # --- 模型实例化和测试 ---
    BATCH_SIZE = 2
    SEQ_LEN = 1
    IMG_C = 3
    IMG_H = 210
    IMG_W = 160

    # 虚拟输入数据
    dummy_obs1 = torch.randn(BATCH_SIZE, SEQ_LEN, IMG_C, IMG_H, IMG_W)
    dummy_actions = torch.randint(0, 18, (BATCH_SIZE, SEQ_LEN))
    dummy_obs2_target = torch.randn(BATCH_SIZE, SEQ_LEN, IMG_C, IMG_H, IMG_W)

    # 增强模型参数
    enc_layers = [2, 3, 4, 3]  # 更深的编码器
    dec_layers = [2, 2, 2, 1]  # 解码器层配置

    model = EnhancedPredictiveRepModel(
        img_in_channels=IMG_C,
        img_out_channels=IMG_C,
        encoder_layers=enc_layers,
        decoder_layers=dec_layers,
        target_img_h=IMG_H,
        target_img_w=IMG_W,
        action_dim=18,
        latent_dim=256,
        base_channels=64,
        num_attention_heads=8,
        use_skip_connections=True
    )

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    # 测试前向传播
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        o_latent, x_reconstructed = model(dummy_obs1, dummy_actions)
    print(f"潜向量 'o_latent' 形状: {o_latent.shape}")
    print(f"重建图像 'x_reconstructed' 形状: {x_reconstructed.shape}")

    # 优化器 - 使用AdamW和学习率调度
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    # 训练设置
    n_epochs = 100
    save_interval = 10
    losses = []

    # 创建保存模型的目录
    save_dir = '../enhanced_model_checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # 训练循环
    print("\nStarting training...")
    model.train()
    for epoch in range(n_epochs):
        loss = model.train_on_batch(dummy_obs1, dummy_actions, dummy_obs2_target, optimizer)
        scheduler.step()  # 更新学习率
        losses.append(loss)

        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.6f}, LR: {current_lr:.2e}')

        # 每10个epoch保存一次模型
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'enhanced_model_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            print(f'Saved enhanced model checkpoint to {checkpoint_path}')

    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'enhanced_model_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'img_in_channels': IMG_C,
            'img_out_channels': IMG_C,
            'encoder_layers': enc_layers,
            'decoder_layers': dec_layers,
            'target_img_h': IMG_H,
            'target_img_w': IMG_W,
            'action_dim': 18,
            'latent_dim': 256,
            'base_channels': 64,
            'num_attention_heads': 8,
            'use_skip_connections': True
        }
    }, final_model_path)
    print(f'Saved final enhanced model to {final_model_path}')

    # 绘制损失曲线
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title('Enhanced Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.yscale('log')

        # 学习率曲线
        plt.subplot(1, 2, 2)
        lrs = [1e-4 * (0.5 ** (epoch // 20)) for epoch in range(n_epochs)]  # 示例学习率
        plt.plot(lrs)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.yscale('log')

        plt.tight_layout()
        plt.savefig('enhanced_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Enhanced training curves saved to enhanced_training_curves.png")
    except ImportError:
        print("Matplotlib not available, skipping loss curve plotting")

    print("\n=== Enhanced Model Summary ===")
    print(f"✅ 总参数数量: {total_params:,}")
    print(f"✅ 编码器层数: {enc_layers} (更深的网络)")
    print(f"✅ 多头注意力: {8} heads")
    print(f"✅ SE注意力块: 已启用")
    print(f"✅ 跳跃连接: 已启用")
    print(f"✅ 梯度裁剪: 已启用")
    print(f"✅ 学习率调度: CosineAnnealing")
    print(f"✅ 权重初始化: Kaiming + Xavier")
    print("Enhanced model training completed!")
