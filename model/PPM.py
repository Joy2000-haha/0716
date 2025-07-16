# predictive_coder_model.py
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
import os
# 假设 .residual 包含 DownBlock2d 和 UpBlock2d。
# 为了独立测试，如果不可用，我们定义虚拟版本。
try:
    from .residual import DownBlock2d, UpBlock2d
    
except ImportError:
    # --- 虚拟/占位残差块 ---
    class DownBlock2d(nn.Module):
        def __init__(self, in_channels, out_channels, num_layers, downsample=False):
            super().__init__()
            self.downsample = downsample
            stride = 2 if downsample else 1
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            # 确保残差连接添加到卷积的输出，而不是块的输入
            self.res_layers = nn.ModuleList()
            for _ in range(num_layers):
                self.res_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
                self.res_layers.append(nn.ReLU(inplace=True)) # 添加激活函数

            # 如果通道数或尺寸不匹配，需要一个 shortcut 连接
            self.shortcut = nn.Identity()
            if stride != 1 or in_channels != out_channels:
                 self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

        def forward(self, x):
            identity = self.shortcut(x) # 计算 shortcut
            x_conv = self.conv(x)
            res_out = x_conv
            for layer in self.res_layers:
                res_out = layer(res_out)
            return res_out + identity # 将卷积后的结果通过残差层，然后与 identity 相加

    class UpBlock2d(nn.Module):
        def __init__(self, in_channels, out_channels, num_layers, upsample=False):
            super().__init__()
            self.upsample_layer = None
            if upsample:
                self.upsample_layer = nn.Upsample(scale_factor=2, mode="nearest")
            
            # 卷积层用于在（可选的）上采样后调整通道数
            # 注意：如果上采样，输入给 conv 的通道数是 in_channels
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            
            self.res_layers = nn.ModuleList()
            for _ in range(num_layers):
                self.res_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
                self.res_layers.append(nn.ReLU(inplace=True))

            # shortcut 连接
            self.shortcut = nn.Identity()
            # UpBlock 通常期望输入通道数等于输出通道数（在主路径的 conv 之后）
            # 但这里的 in_channels 是给 self.conv 的。如果 self.conv 改变了通道数，
            # 并且我们希望 shortcut 基于 self.conv 的输入，需要调整。
            # 通常，UpBlock 中的 shortcut 会来自编码器的对应层 (skip connection)，这里简化了。
            # 为了简化，我们假设残差是基于 self.conv 的输出。
            # 如果 self.conv 更改了通道数，那么 identity (来自 self.conv(x)) 需要调整
            # 为确保维度匹配， shortcut 通常会先将输入 x (可能已上采样) 通过一个 1x1 卷积。
            # 但在此简化版本中，我们只对 self.conv 的输出应用残差。


        def forward(self, x, skip):  # skip 连接在此简化虚拟版本中未使用
            if self.upsample_layer:
                x = self.upsample_layer(x)
            
            x_conv = self.conv(x) # 主路径卷积
            identity = x_conv # 残差连接的基础是主路径卷积的输出

            res_out = x_conv
            for layer in self.res_layers:
                res_out = layer(res_out)
            return res_out + identity # 残差相加

# --- 编码器 ---
class Encoder(nn.Module):
    def __init__(self, in_channels: int, layers: List[int]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)  # H/2, W/2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # H/4, W/4
        self.down1 = DownBlock2d(64, 64, layers[0])
        self.down2 = DownBlock2d(64, 128, layers[1], downsample=True)  # H/8, W/8

        # 假设输入为 210x160 (C, H, W 经过预处理后)
        # H_conv1 = floor((210 - 7 + 2*3)/2) + 1 = 105
        # W_conv1 = floor((160 - 7 + 2*3)/2) + 1 = 80
        # H_pool = floor((105 - 3 + 2*1)/2) + 1 = 53
        # W_pool = floor((80 - 3 + 2*1)/2) + 1 = 40
        # H_down1 = 53 (DownBlock2d 中的卷积 stride=1)
        # W_down1 = 40
        # H_down2: DownBlock2d 中 downsample=True, 第一个卷积 stride=2
        #   H_after_conv_in_down2 = floor((53 - 3 + 2*1)/2)+1 = 27
        #   W_after_conv_in_down2 = floor((40 - 3 + 2*1)/2)+1 = 20
        
        self.final_C_before_flatten = 128 # down2 的输出通道
        self.final_H_before_flatten = 27 # 对于 210 输入 H，down2 的输出 H
        self.final_W_before_flatten = 20 # 对于 160 输入 W，down2 的输出 W

        self.flatten = nn.Flatten(start_dim=1) # 从通道维度开始展平
        self.to_256_vector = nn.Linear(
            self.final_C_before_flatten * self.final_H_before_flatten * self.final_W_before_flatten,
            256 # 目标向量维度
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.down1(x)
        x = self.down2(x)  # 输出: (B_effective, 128, 27, 20)
        x = self.flatten(x)
        x = self.to_256_vector(x) # 输出: (B_effective, 256)
        return x

# --- 解码器 ---
class Decoder(nn.Module):
    def __init__(self, in_channels_latent: int, out_channels_img: int, layers: List[int]) -> None:
        super().__init__()
        # in_channels_latent 是指输入到第一个上采样块的特征图通道数 (例如128)
        self.up1 = UpBlock2d(in_channels_latent, 64, layers[0], upsample=True) # 27x20 -> 54x40 (假设)
        self.up2 = UpBlock2d(64, 64, layers[1], upsample=True)                 # 54x40 -> 108x80
        self.up3 = UpBlock2d(64, 64, layers[2] if len(layers) > 2 else 1, upsample=True) # 108x80 -> 216x160
        
        # 最后的转置卷积调整通道数到目标图像通道，并可能微调尺寸
        # 这里用 ConvTranspose2d 是为了匹配原始结构，kernel_size=1, stride=1 通常不改变空间维度
        self.conv_final = nn.ConvTranspose2d(64, out_channels_img, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x, None) # skip 参数未使用
        x = self.up2(x, None)
        x = self.up3(x, None)
        x = self.conv_final(x) # 输出: (B_effective, out_channels_img, H_decoder, W_decoder)
        return x

# --- 预测编码器模型 ---
class PredictiveRepModel(nn.Module):
    def __init__(
            self, img_in_channels: int, img_out_channels: int, 
            encoder_layers: List[int], decoder_layers_config: List[int], # decoder_layers_config 是给Decoder的层数列表
            target_img_h: int, target_img_w: int # 目标图像的 H, W (用于裁剪)
    ) -> None:
        super().__init__()
        self.encoder = Encoder(img_in_channels, encoder_layers)

        # 通道扩展器：将 256 维向量复制到 18 个通道，每个通道都是这个向量
        self.channel_expander = nn.Conv2d(in_channels=1, out_channels=18, kernel_size=1, bias=False)
        self.channel_expander.weight.data.fill_(1.0) # 权重初始化为1实现复制

        # 解码器输入参数
        self.decoder_input_channels = 128 # 第一个 UpBlock 的输入通道数 (来自 encoder.final_C_before_flatten)
        self.decoder_input_height = self.encoder.final_H_before_flatten # 27
        self.decoder_input_width = self.encoder.final_W_before_flatten  # 20

        # 线性层：将掩码后的 256 维向量投影回解码器期望的 (C*H*W) 维度
        self.project_to_decoder_input = nn.Linear(
            256, # 输入维度
            self.decoder_input_channels * self.decoder_input_height * self.decoder_input_width
        )
        
        self.decoder = Decoder(self.decoder_input_channels, img_out_channels, decoder_layers_config)

        self.target_img_h = target_img_h # 存储目标图像高度
        self.target_img_w = target_img_w # 存储目标图像宽度

        # 初始化权重
        self.apply(self._init_weights)


    def _init_weights(self, m): # 权重初始化辅助函数
        if isinstance(m, nn.Conv2d):
            if m != self.channel_expander: # 不重新初始化 channel_expander
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

    def forward(self, x_obs1: torch.Tensor, actions: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        前向传播。
        参数:
            x_obs1 (torch.Tensor): 输入观测 obs1，形状 (B, L, C_in, H_in, W_in)。L 通常为 1。
            actions (torch.Tensor): 动作张量，形状 (B, L)。L 通常为 1。
        返回:
            o_latent (torch.Tensor): 编码器输出的潜向量，形状 (B, L, 256)。
            output_x_reconstructed (torch.Tensor): 解码器重建的下一帧观测 obs2_pred，
                                                 形状 (B, L, C_out, H_decoder, W_decoder)。
        """
        B, L, C_in, H_in, W_in = x_obs1.shape
        # 将输入 reshape 以便编码器处理: (B*L, C_in, H_in, W_in)
        x_reshaped = x_obs1.reshape(B * L, C_in, H_in, W_in)

        # 1. 编码图像到 256 维向量
        encoded_vector = self.encoder(x_reshaped)  # 输出形状: (B*L, 256)
        
        # 2. 通道扩展：为卷积准备，(B*L, 256) -> (B*L, 1, 256, 1)
        vec_for_conv = encoded_vector.unsqueeze(1).unsqueeze(-1)
        expanded_18_channels = self.channel_expander(vec_for_conv)  # 输出形状: (B*L, 18, 256, 1)

        # 3. 动作掩码：根据动作选择特定通道
        action_indices = actions.reshape(B * L).long() # 确保动作为 (B*L,) 的长整型张量
        
        if not torch.all((action_indices >= 0) & (action_indices < 18)):
            raise ValueError(f"所有动作索引必须在 [0, 17] 范围内。收到: {action_indices}")

        # 使用 gather 选择通道：idx 形状 (B*L, 1, 1, 1)
        idx = action_indices.view(B * L, 1, 1, 1)
        # 扩展 idx 以匹配 expanded_18_channels 的后两个维度进行 gather
        idx = idx.repeat(1, 1, expanded_18_channels.size(2), expanded_18_channels.size(3))
        
        masked_vector_channel = torch.gather(expanded_18_channels, 1, idx) # 输出形状: (B*L, 1, 256, 1)
        
        # 4. 准备解码器输入
        masked_flat = masked_vector_channel.reshape(B * L, -1)  # 展平: (B*L, 256)
        decoder_input_latent = self.project_to_decoder_input(masked_flat) # 投影到解码器输入维度
        decoder_input_reshaped = decoder_input_latent.view(
            B * L,
            self.decoder_input_channels,
            self.decoder_input_height,
            self.decoder_input_width
        ) # Reshape: (B*L, D_in_C, D_in_H, D_in_W)

        # 5. 解码
        decoded_frames_raw = self.decoder(decoder_input_reshaped) # 输出: (B*L, C_out, H_decoder, W_decoder)

        # 将解码器输出 reshape回 (B, L, C_out, H_decoder, W_decoder)
        final_out_C, final_out_H, final_out_W = decoded_frames_raw.shape[1:]
        output_x_reconstructed = decoded_frames_raw.reshape(B, L, final_out_C, final_out_H, final_out_W)

        # 将编码器输出 reshape 回 (B, L, 256)
        o_latent_return = encoded_vector.reshape(B, L, 256)

        return o_latent_return, output_x_reconstructed

    def _crop_output(self, predicted_obs_raw: torch.Tensor) -> torch.Tensor:
        """
        裁剪模型输出以匹配目标图像尺寸 (self.target_img_h, self.target_img_w)。
        输入 predicted_obs_raw: (B, C, H_decoder, W_decoder)
        """
        current_h, current_w = predicted_obs_raw.shape[2], predicted_obs_raw.shape[3]
        h_slice = slice(0, min(current_h, self.target_img_h))
        w_slice = slice(0, min(current_w, self.target_img_w))
        
        cropped_obs = predicted_obs_raw[:, :, h_slice, w_slice]
        
        # 如果需要，可以添加填充逻辑，但通常裁剪就足够了
        # if cropped_obs.shape[2] < self.target_img_h or cropped_obs.shape[3] < self.target_img_w:
        #     # 进行填充 (这里简化，通常不期望需要填充预测)
        #     pass
        return cropped_obs

    def train_on_batch(self, obs1_input_b_l_c_h_w: torch.Tensor, 
                       actions_input_b_l: torch.Tensor, 
                       obs2_target_b_l_c_h_w: torch.Tensor, 
                       optimizer: torch.optim.Optimizer) -> float:
        """
        在单个批次上训练模型。
        参数:
            obs1_input_b_l_c_h_w: obs1批次, 形状 (B, L, C, H, W)
            actions_input_b_l: 动作批次, 形状 (B, L)
            obs2_target_b_l_c_h_w: obs2目标批次, 形状 (B, L, C, H, W)
            optimizer: PyTorch优化器
        返回:
            loss_value (float): 当前批次的损失值
        """
        self.train() # 设置为训练模式
        optimizer.zero_grad() # 清空梯度

        # 前向传播
        _, predicted_obs2_raw_b_l_c_hd_wd = self.forward(obs1_input_b_l_c_h_w, actions_input_b_l)
        
        # 假设 L=1, 移除 L 维度进行损失计算和裁剪
        # predicted_obs2_raw: (B, C, H_decoder, W_decoder)
        # obs2_target_for_loss: (B, C, H, W)
        predicted_obs2_processed_b_c_hd_wd = predicted_obs2_raw_b_l_c_hd_wd.squeeze(1) 
        obs2_target_for_loss_b_c_h_w = obs2_target_b_l_c_h_w.squeeze(1)

        # 裁剪预测输出以匹配目标尺寸
        predicted_obs2_cropped_b_c_h_w = self._crop_output(predicted_obs2_processed_b_c_hd_wd)
        
        # 计算损失 (MSE)
        loss = F.mse_loss(predicted_obs2_cropped_b_c_h_w, obs2_target_for_loss_b_c_h_w)
        
        loss.backward() # 反向传播
        optimizer.step() # 更新权重
        
        return loss.item()

    def evaluate_on_batch(self, obs1_input_b_l_c_h_w: torch.Tensor, 
                          actions_input_b_l: torch.Tensor, 
                          obs2_target_b_l_c_h_w: torch.Tensor) -> float:
        """
        在单个批次上评估模型。
        参数:
            (与 train_on_batch 类似)
        返回:
            loss_value (float): 当前批次的损失值
        """
        self.eval() # 设置为评估模式
        with torch.no_grad(): # 不计算梯度
            _, predicted_obs2_raw_b_l_c_hd_wd = self.forward(obs1_input_b_l_c_h_w, actions_input_b_l)
            
            predicted_obs2_processed_b_c_hd_wd = predicted_obs2_raw_b_l_c_hd_wd.squeeze(1)
            obs2_target_for_loss_b_c_h_w = obs2_target_b_l_c_h_w.squeeze(1)

            predicted_obs2_cropped_b_c_h_w = self._crop_output(predicted_obs2_processed_b_c_hd_wd)
            
            loss = F.mse_loss(predicted_obs2_cropped_b_c_h_w, obs2_target_for_loss_b_c_h_w)
            
        return loss.item()


if __name__ == '__main__':
    # --- 模型实例化和测试 ---
    BATCH_SIZE = 2
    SEQ_LEN = 1 # 对于 obs1 -> obs2 预测, L=1
    IMG_C = 3
    IMG_H = 210
    IMG_W = 160
    
    # 虚拟输入数据
    dummy_obs1 = torch.randn(BATCH_SIZE, SEQ_LEN, IMG_C, IMG_H, IMG_W)
    dummy_actions = torch.randint(0, 18, (BATCH_SIZE, SEQ_LEN)) # 动作索引
    dummy_obs2_target = torch.randn(BATCH_SIZE, SEQ_LEN, IMG_C, IMG_H, IMG_W) # 目标下一帧

    # 模型参数
    enc_layers = [2, 2] # 编码器中每个DownBlock的层数
    # 解码器中每个UpBlock的层数，通常与编码器对称或自定义
    # Decoder 有3个 UpBlock (up1, up2, up3)，所以需要3个值
    dec_layers_cfg = [enc_layers[1], enc_layers[0], 1] # 例如: [2, 2, 1]

    model = PredictiveCoder(
        img_in_channels=IMG_C,
        img_out_channels=IMG_C, # 输出通道数与输入一致
        encoder_layers=enc_layers,
        decoder_layers_config=dec_layers_cfg,
        target_img_h=IMG_H, # 目标裁剪高度
        target_img_w=IMG_W  # 目标裁剪宽度
    )
    # 训练设置
    n_epochs = 100
    save_interval = 10
    losses = []
    
    # 创建保存模型的目录
    save_dir = 'model_checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练循环
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
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    # 测试前向传播
    # model.eval()
    # with torch.no_grad():
    #     o_latent, x_reconstructed = model(dummy_obs1, dummy_actions)
    # print(f"潜向量 'o_latent' 形状: {o_latent.shape}")  # 期望: (B, L, 256)
    # print(f"重建图像 'x_reconstructed' 形状: {x_reconstructed.shape}") # 期望: (B, L, C_out, H_decoder, W_decoder)

    # # 测试训练一个批次
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # loss_train = model.train_on_batch(dummy_obs1, dummy_actions, dummy_obs2_target, optimizer)
    # print(f"训练批次损失: {loss_train:.6f}")

    # # 测试评估一个批次
    # loss_eval = model.evaluate_on_batch(dummy_obs1, dummy_actions, dummy_obs2_target)
    # print(f"评估批次损失: {loss_eval:.6f}")