# predictive_coder_model.py
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
import os
import math

# 假设 .residual 包含 DownBlock2d 和 UpBlock2d。
# 为了独立测试，如果不可用，我们定义虚拟版本。
try:
    from .residual import DownBlock2d, UpBlock2d
    
except ImportError:
    # --- 残差块实现 ---
    # --- 残差块实现 ---
    class ResidualBlock(nn.Module):
        """基本残差块"""
        def __init__(self, channels, num_layers=2):
            super().__init__()
            self.layers = nn.ModuleList()
            for i in range(num_layers):
                self.layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
                if i < num_layers - 1:  # 最后一层不加激活函数
                    self.layers.append(nn.BatchNorm2d(channels))
                    self.layers.append(nn.ReLU()) # <--- 修改点
            
            # 最后的批归一化
            self.final_bn = nn.BatchNorm2d(channels)
            self.final_relu = nn.ReLU() # <--- 修改点

        def forward(self, x):
            identity = x
            out = x
            
            for layer in self.layers:
                out = layer(out)
            
            out = self.final_bn(out)
            # out += identity  # 残差连接 # 这是旧代码
            out = out + identity # <--- 修改点：改为非原地加法
            out = self.final_relu(out)
            return out

    class DownBlock2d(nn.Module):
        def __init__(self, in_channels, out_channels, num_layers, downsample=False):
            super().__init__()
            self.downsample = downsample
            stride = 2 if downsample else 1
            
            # 主卷积层
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU() # <--- 修改点
            
            # 残差块
            self.residual_blocks = nn.ModuleList([
                ResidualBlock(out_channels, 2) for _ in range(num_layers) # ResidualBlock 内部也需要按上述修改
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
            
            if self.downsample and identity.shape[2:] != out.shape[2:]:
                identity = F.adaptive_avg_pool2d(identity, out.shape[2:])
            
            # out += identity  # 残差连接 # 这是旧代码
            out = out + identity # <--- 修改点：改为非原地加法
            
            for res_block in self.residual_blocks:
                out = res_block(out)
                
            return out

    class UpBlock2d(nn.Module):
        def __init__(self, in_channels, out_channels, num_layers, upsample=False):
            super().__init__()
            self.upsample_layer = None
            if upsample:
                self.upsample_layer = nn.Upsample(scale_factor=2, mode="nearest")
            
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
            
            # 残差块
            self.residual_blocks = nn.ModuleList([
                ResidualBlock(out_channels, 2) for _ in range(num_layers)
            ])

        def forward(self, x, skip=None):  # skip连接在此简化版本中未使用
            if self.upsample_layer:
                x = self.upsample_layer(x)
            
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)

            # 通过残差块
            for res_block in self.residual_blocks:
                x = res_block(x)
                
            return x

# --- 注意力机制 ---
class ActionAttention(nn.Module):
    """基于动作的注意力机制"""
    def __init__(self, action_dim=18, latent_dim=256, hidden_dim=128):
        super().__init__()
        
        # 动作编码器：one-hot -> 三层全连接 -> softmax
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Softmax(dim=-1)  # 生成注意力权重
        )
        
        # 用于生成K, V的线性层
        self.key_proj = nn.Linear(latent_dim, latent_dim)
        self.value_proj = nn.Linear(latent_dim, latent_dim)
        
        # 缩放因子
        self.scale = math.sqrt(latent_dim)
        
    def forward(self, encoded_features, actions):
        """
        Args:
            encoded_features: 编码器输出 (B, latent_dim)
            actions: 动作索引 (B,)
        Returns:
            attended_features: 注意力加权后的特征 (B, latent_dim)
        """
        batch_size = encoded_features.size(0)
        
        # 将动作转换为one-hot编码
        actions_one_hot = F.one_hot(actions.long(), num_classes=18).float()  # (B, 18)
        
        # 通过动作编码器生成查询向量Q
        Q = self.action_encoder(actions_one_hot)  # (B, latent_dim)
        
        # 从编码特征生成K, V
        K = self.key_proj(encoded_features)    # (B, latent_dim)
        V = self.value_proj(encoded_features)  # (B, latent_dim)
        
        # 计算注意力权重
        # Q: (B, latent_dim), K: (B, latent_dim)
        attention_scores = torch.sum(Q * K, dim=-1, keepdim=True) / self.scale  # (B, 1)
        attention_weights = torch.softmax(attention_scores, dim=0)  # 在批次维度上softmax
        
        # 应用注意力权重
        attended_features = attention_weights * V  # (B, latent_dim)
        
        return attended_features

# --- 编码器 ---
class Encoder(nn.Module):
    def __init__(self, in_channels: int, layers: List[int]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)  # H/2, W/2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # H/4, W/4
        
        # 使用带残差连接的下采样块
        self.down1 = DownBlock2d(64, 64, layers[0])
        self.down2 = DownBlock2d(64, 128, layers[1], downsample=True)  # H/8, W/8

        # 计算最终特征图尺寸
        self.final_C_before_flatten = 128
        self.final_H_before_flatten = 27  # 对于210输入H
        self.final_W_before_flatten = 20  # 对于160输入W

        self.flatten = nn.Flatten(start_dim=1)
        self.to_256_vector = nn.Linear(
            self.final_C_before_flatten * self.final_H_before_flatten * self.final_W_before_flatten,
            256
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.down1(x)
        x = self.down2(x)  # 输出: (B_effective, 128, 27, 20)
        x = self.flatten(x)
        x = self.to_256_vector(x)  # 输出: (B_effective, 256)
        return x

# --- 解码器 ---
class Decoder(nn.Module):
    def __init__(self, in_channels_latent: int, out_channels_img: int, layers: List[int]) -> None:
        super().__init__()
        self.up1 = UpBlock2d(in_channels_latent, 64, layers[0], upsample=True)
        self.up2 = UpBlock2d(64, 64, layers[1], upsample=True)
        self.up3 = UpBlock2d(64, 64, layers[2] if len(layers) > 2 else 1, upsample=True)
        
        self.conv_final = nn.ConvTranspose2d(64, out_channels_img, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x, None)
        x = self.up2(x, None)
        x = self.up3(x, None)
        x = self.conv_final(x)
        return x

# --- 预测编码器模型 ---
class PredictiveRepModel(nn.Module):
    def __init__(
            self, img_in_channels: int, img_out_channels: int, 
            encoder_layers: List[int], decoder_layers_config: List[int],
            target_img_h: int, target_img_w: int,
            action_dim: int = 18, latent_dim: int = 256
    ) -> None:
        super().__init__()
        self.encoder = Encoder(img_in_channels, encoder_layers)

        # 注意力机制替代原来的掩码机制
        self.attention = ActionAttention(action_dim=action_dim, latent_dim=latent_dim)

        # 解码器输入参数
        self.decoder_input_channels = 128
        self.decoder_input_height = self.encoder.final_H_before_flatten  # 27
        self.decoder_input_width = self.encoder.final_W_before_flatten   # 20

        # 线性层：将注意力输出投影回解码器期望的维度
        self.project_to_decoder_input = nn.Linear(
            latent_dim,
            self.decoder_input_channels * self.decoder_input_height * self.decoder_input_width
        )
        
        self.decoder = Decoder(self.decoder_input_channels, img_out_channels, decoder_layers_config)

        self.target_img_h = target_img_h
        self.target_img_w = target_img_w

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
             nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
             if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_obs1: torch.Tensor, actions: torch.Tensor):
        """
        前向传播。
        参数:
            x_obs1 (torch.Tensor): 输入观测 obs1，形状 (B, L, C_in, H_in, W_in)。L 通常为 1。
            actions (torch.Tensor): 动作张量，形状 (B, L)。L 通常为 1。
        返回:
            o_latent (torch.Tensor): 编码器输出的潜向量，形状 (B, L, 256)。
            output_x_reconstructed (torch.Tensor): 解码器重建的下一帧观测，
                                                 形状 (B, L, C_out, H_decoder, W_decoder)。
        """
        B, L, C_in, H_in, W_in = x_obs1.shape
        # 将输入 reshape 以便编码器处理: (B*L, C_in, H_in, W_in)
        x_reshaped = x_obs1.reshape(B * L, C_in, H_in, W_in)
        actions_reshaped = actions.reshape(B * L)

        # 1. 编码图像到 256 维向量
        encoded_vector = self.encoder(x_reshaped)  # 输出形状: (B*L, 256)
        
        # 2. 使用注意力机制处理编码特征和动作
        attended_features = self.attention(encoded_vector, actions_reshaped)  # (B*L, 256)
        
        # 3. 准备解码器输入
        decoder_input_latent = self.project_to_decoder_input(attended_features)
        decoder_input_reshaped = decoder_input_latent.view(
            B * L,
            self.decoder_input_channels,
            self.decoder_input_height,
            self.decoder_input_width
        )

        # 4. 解码
        decoded_frames_raw = self.decoder(decoder_input_reshaped)

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
        """在单个批次上训练模型"""
        self.train()
        optimizer.zero_grad()

        # 前向传播
        _, predicted_obs2_raw_b_l_c_hd_wd = self.forward(obs1_input_b_l_c_h_w, actions_input_b_l)
        
        # 处理预测输出和目标
        predicted_obs2_processed_b_c_hd_wd = predicted_obs2_raw_b_l_c_hd_wd.squeeze(1) 
        obs2_target_for_loss_b_c_h_w = obs2_target_b_l_c_h_w.squeeze(1)

        # 裁剪预测输出以匹配目标尺寸
        predicted_obs2_cropped_b_c_h_w = self._crop_output(predicted_obs2_processed_b_c_hd_wd)
       # print(f"Predicted shape: {predicted_obs2_cropped_b_c_h_w.shape}, Target shape: {obs2_target_for_loss_b_c_h_w.shape}")
        # 计算损失
        loss = F.mse_loss(predicted_obs2_cropped_b_c_h_w, obs2_target_for_loss_b_c_h_w)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def evaluate_on_batch(self, obs1_input_b_l_c_h_w: torch.Tensor, 
                          actions_input_b_l: torch.Tensor, 
                          obs2_target_b_l_c_h_w: torch.Tensor) -> float:
        """在单个批次上评估模型"""
        self.eval()
        with torch.no_grad():
            _, predicted_obs2_raw_b_l_c_hd_wd = self.forward(obs1_input_b_l_c_h_w, actions_input_b_l)
            
            predicted_obs2_processed_b_c_hd_wd = predicted_obs2_raw_b_l_c_hd_wd.squeeze(1)
            obs2_target_for_loss_b_c_h_w = obs2_target_b_l_c_h_w.squeeze(1)

            predicted_obs2_cropped_b_c_h_w = self._crop_output(predicted_obs2_processed_b_c_hd_wd)
            
            loss = F.mse_loss(predicted_obs2_cropped_b_c_h_w, obs2_target_for_loss_b_c_h_w)
            
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

    # 模型参数
    enc_layers = [2, 2]
    dec_layers_cfg = [2, 2, 1]

    model = PredictiveRepModel(
        img_in_channels=IMG_C,
        img_out_channels=IMG_C,
        encoder_layers=enc_layers,
        decoder_layers_config=dec_layers_cfg,
        target_img_h=IMG_H,
        target_img_w=IMG_W
    )
    
    # 测试前向传播
    print("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        o_latent, x_reconstructed = model(dummy_obs1, dummy_actions)
    print(f"潜向量 'o_latent' 形状: {o_latent.shape}")
    print(f"重建图像 'x_reconstructed' 形状: {x_reconstructed.shape}")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练设置
    n_epochs = 100
    save_interval = 10
    losses = []
    
    # 创建保存模型的目录
    save_dir = 'model_checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练循环
    print("\nStarting training...")
    model.train()
    for epoch in range(n_epochs):
        loss = model.train_on_batch(dummy_obs1, dummy_actions, dummy_obs2_target, optimizer)
        losses.append(loss)
        
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss:.6f}')
        
        # 每10个epoch保存一次模型
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved model checkpoint to {checkpoint_path}')
    
    # 绘制损失曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('training_loss.png')
        plt.close()
        print("Loss curve saved to training_loss.png")
    except ImportError:
        print("Matplotlib not available, skipping loss curve plotting")