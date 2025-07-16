# AtariDataLoder.py
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
# import cv2
import random
import logging
import tempfile
import shutil

logger = logging.getLogger(__name__)


class AtariEpisode:
    """单个Atari游戏episode的数据结构"""
    
    def __init__(self, pkl_path: str, args):
        self.pkl_path = pkl_path
        self.args = args
        self.data = None
        self.length = 0
        
        self._load_episode()
        
    def _load_episode(self):
        """从PKL文件加载episode数据"""
        try:
            with open(self.pkl_path, 'rb') as f:
                self.data = pickle.load(f)

            # 处理 'done' 键的不同命名
            if 'done' not in self.data:
                if 'terminals' in self.data:
                    logger.debug(f"Using 'terminals' as 'done' in {self.pkl_path}")
                    self.data['done'] = self.data['terminals']
                elif 'terminal' in self.data:
                    logger.debug(f"Using 'terminal' as 'done' in {self.pkl_path}")
                    self.data['done'] = self.data['terminal']
                else:
                    raise KeyError(f"Missing required key 'done' (or 'terminals'/'terminal') in {self.pkl_path}")
            
            # 检查必需的键
            required_keys = ['observations', 'actions', 'done']
            missing_keys = [key for key in required_keys if key not in self.data]
            if missing_keys:
                raise KeyError(f"Missing required keys {missing_keys} in {self.pkl_path}")
            
            self.observations = self.data['observations']
            self.actions = self.data['actions']
            self.done = self.data['done']
            
            self.length = len(self.observations)
            
            # 验证数据一致性
            if not (len(self.observations) == len(self.actions) == len(self.done)):
                raise ValueError(f"Data length mismatch in {self.pkl_path}: "
                               f"obs={len(self.observations)}, "
                               f"actions={len(self.actions)}, "
                               f"done={len(self.done)}")
            
            logger.debug(f"Loaded episode {self.pkl_path}: {self.length} steps")
            
        except Exception as e:
            logger.error(f"Failed to load {self.pkl_path}: {e}")
            raise
            
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """预处理单张图像"""
        target_size = (self.args.img_width, self.args.img_height)
        if img.shape[:2] != (self.args.img_height, self.args.img_width):
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
                
        # 转换为(C, H, W)格式
        img = np.transpose(img, (2, 0, 1))
        
        return img
        
    def get_sequence(self, start_idx: int) -> Dict[str, np.ndarray]:
        """获取从start_idx开始的序列数据"""
        seq_len = self.args.sequence_length + 1
        end_idx = start_idx + seq_len
        
        obs_seq = self.observations[start_idx:end_idx]
        actions_seq = self.actions[start_idx:end_idx]
        done_seq = self.done[start_idx:end_idx]
        
        processed_obs = np.stack([self._preprocess_image(obs) for obs in obs_seq])
        
        return {
            'observations': processed_obs,
            'actions': actions_seq,
            'done': done_seq
        }


class AtariDataset(Dataset):
    """Atari数据集"""
    
    def __init__(self, args, pkl_files: List[str]):
        self.args = args
        self.episode_paths = pkl_files
        self.sequence_indices = []
        
        self._build_sequence_indices()
        
    def _build_sequence_indices(self):
        """构建序列索引"""
        for ep_idx, pkl_path in enumerate(self.episode_paths):
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                    ep_length = len(data['observations'])
            except Exception as e:
                logger.warning(f"Could not read length from {pkl_path}: {e}")
                continue
                
            seq_len = self.args.sequence_length + 1
            if ep_length < seq_len:
                logger.debug(f"Episode {pkl_path} too short ({ep_length} < {seq_len}), skipping")
                continue
                
            # 创建重叠的序列
            for start_idx in range(0, ep_length - seq_len + 1, self.args.overlap_steps):
                self.sequence_indices.append((ep_idx, start_idx))
                    
        logger.info(f"Generated {len(self.sequence_indices)} sequences from {len(self.episode_paths)} episodes")
        
    def __len__(self) -> int:
        return len(self.sequence_indices)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, start_idx = self.sequence_indices[idx]
        pkl_path = self.episode_paths[ep_idx]
        
        # 每次都从文件加载（简化缓存逻辑）
        episode = AtariEpisode(pkl_path, self.args)
        sequence_data = episode.get_sequence(start_idx)
        
        return {
            'observations': torch.from_numpy(sequence_data['observations']).float(),
            'actions': torch.from_numpy(sequence_data['actions']).long(),
            'done': torch.from_numpy(sequence_data['done']).bool()
        }


def create_atari_dataloaders(args) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """
    创建Atari数据加载器
    
    Args:
        args: 包含所有必要参数的命名空间对象，需要包含：
            - data_dir: 数据目录路径
            - sequence_length: 序列长度
            - batch_size: 批次大小
            - overlap_steps: 重叠步数
            - num_workers: 工作进程数
            - train_split: 训练集比例
            - val_split: 验证集比例
            - img_height, img_width: 图像尺寸
            - max_episodes (可选): 最大episode数量
        
    Returns:
        Tuple[train_loader, val_loader]: 训练和验证数据加载器
    """
    if not os.path.isdir(args.data_dir):
        logger.error(f"Data directory '{args.data_dir}' does not exist.")
        return None, None
        
    pkl_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.pkl')]
    if not pkl_files:
        logger.error(f"No .pkl files found in '{args.data_dir}'.")
        return None, None

    # 随机打乱文件列表
    random.shuffle(pkl_files)
    
    # 限制episode数量（用于调试）
    if hasattr(args, 'max_episodes') and args.max_episodes is not None:
        pkl_files = pkl_files[:args.max_episodes]
        logger.info(f"Limited to {len(pkl_files)} episodes for debugging")

    # 划分数据集
    train_split_idx = int(len(pkl_files) * args.train_split)
    val_split_idx = train_split_idx + int(len(pkl_files) * args.val_split)
    
    train_files = pkl_files[:train_split_idx]
    val_files = pkl_files[train_split_idx:val_split_idx]
    
    logger.info(f"Dataset split: {len(train_files)} train, {len(val_files)} val, "
                f"{len(pkl_files) - val_split_idx} test episodes")

    # 创建数据集
    train_dataset = AtariDataset(args, train_files)
    val_dataset = AtariDataset(args, val_files) if len(val_files) > 0 else None

    if len(train_dataset) == 0:
        logger.error("Training dataset is empty after processing. Check sequence_length and overlap_steps.")
        return None, None
    
    # 数据加载器通用参数
    loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': True,
        'drop_last': True,
    }
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs) if val_dataset else None
    
    return train_loader, val_loader


# === 测试工具函数 ===
def _create_dummy_data_for_test(data_dir: str, num_files: int, num_steps: int):
    """为测试创建虚拟数据"""
    logger.info(f"Creating dummy data for test in: {data_dir}")
    os.makedirs(data_dir, exist_ok=True)
    for i in range(num_files):
        dummy_episode = {
            'observations': np.random.randint(0, 255, (num_steps, 210, 160, 3), dtype=np.uint8),
            'actions': np.random.randint(0, 18, num_steps, dtype=np.int64),
            'done': np.random.choice([True, False], num_steps, p=[0.01, 0.99]),
        }
        with open(os.path.join(data_dir, f'dummy_episode_{i}.pkl'), 'wb') as f:
            pickle.dump(dummy_episode, f)


def test_dataloader():
    """测试数据加载器"""
    # 创建一个临时目录进行测试
    test_data_dir = "../Data/replaydata"
    
    try:
        # 创建虚拟参数对象
        class Args:
            def __init__(self):
                self.data_dir = test_data_dir
                self.sequence_length = 8
                self.batch_size = 4
                self.overlap_steps = 4
                self.num_workers = 0
                self.train_split = 0.8
                self.val_split = 0.2
                self.img_height = 210
                self.img_width = 160
                self.max_episodes = 3
        
        args = Args()
        
        # 创建测试数据
        _create_dummy_data_for_test(test_data_dir, num_files=5, num_steps=100)
        
        # 测试数据加载器
        train_loader, val_loader = create_atari_dataloaders(args)
        
        assert train_loader is not None, "Train loader creation failed"
        
        logger.info("=== 数据加载器测试 ===")
        
        # 测试一个批次
        batch = next(iter(train_loader))
        
        expected_obs_shape = (args.batch_size, args.sequence_length + 1, 3, args.img_height, args.img_width)
        assert batch['observations'].shape == expected_obs_shape, \
            f"Obs shape mismatch: Got {batch['observations'].shape}, Expected {expected_obs_shape}"
        
        logger.info(f"  observations: {batch['observations'].shape}")
        logger.info(f"  actions: {batch['actions'].shape}")
        logger.info(f"  done: {batch['done'].shape}")
        
        # 测试数据类型和值范围
        assert batch['observations'].dtype == torch.float32
        assert batch['actions'].dtype == torch.int64
        assert batch['done'].dtype == torch.bool
        assert 0.0 <= batch['observations'].min() <= batch['observations'].max() <= 1.0
        
        logger.info("✅ 数据加载器测试成功!")
        
    except Exception as e:
        logger.error(f"❌ 数据加载器测试失败: {e}", exc_info=True)
        raise
    finally:
        # 清理临时目录
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    test_dataloader()