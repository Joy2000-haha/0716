# ReplayBuffer.py
import gymnasium as gym
import ale_py
import numpy as np
import pickle
import os
import argparse
import json
from typing import List, Tuple, Optional
from datetime import datetime
import logging
import multiprocessing as mp
from multiprocessing import Pool, Manager
import time

gym.register_envs(ale_py)  # 注册 Atari 环境


class AtariDataGenerator:
    """
    用于生成 Atari 游戏回放数据的类
    """
    
    def __init__(self, 
                 num_plats: int = 5,
                 num_episodes_every_plat: int = 10,
                 game_ratios: Optional[List[float]] = None,
                 render_mode: Optional[str] = None,
                 save_dir: str = "replay_data",
                 custom_games: Optional[List[str]] = None,
                 process_id: int = 0):
        """
        初始化数据生成器
        
        参数:
            num_plats (int): 并行环境数量
            num_episodes_every_plat (int): 每个环境运行的回合数
            game_ratios (List[float], optional): 各游戏类型的比例，默认平均分配
            render_mode (str, optional): 渲染模式
            save_dir (str): 数据保存目录
            custom_games (List[str], optional): 自定义游戏列表
            process_id (int): 进程ID，用于区分不同进程
        """
        self.num_plats = num_plats
        self.num_episodes_every_plat = num_episodes_every_plat
        self.save_dir = save_dir
        self.render_mode = render_mode
        self.process_id = process_id
        
        # 设置可用游戏列表
        if custom_games:
            self.available_games = custom_games
        else:
            # 默认游戏列表（可扩展）
            self.available_games = [
                "Seaquest-v4",
                "Riverraid-v4", 
                "ChopperCommand-v4"
            ]
        
        print(f"进程 {self.process_id}: 使用游戏列表: {self.available_games}")
        
        # 设置游戏比例
        if game_ratios is None:
            self.game_ratios = [1.0 / len(self.available_games)] * len(self.available_games)
        else:
            assert len(game_ratios) == len(self.available_games), f"游戏比例数量({len(game_ratios)})必须与游戏数量({len(self.available_games)})一致"
            assert abs(sum(game_ratios) - 1.0) < 1e-6, "游戏比例总和必须为1"
            self.game_ratios = game_ratios
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 计算每种游戏的环境数量
        self._allocate_games()
        
        # 初始化环境
        self._initialize_environments()
        
        # 存储所有回放数据
        self.replay_data = []
        
    def _allocate_games(self):
        """根据比例分配各游戏的环境数量"""
        self.game_allocation = []
        remaining_plats = self.num_plats
        
        for i, ratio in enumerate(self.game_ratios[:-1]):
            count = int(self.num_plats * ratio)
            self.game_allocation.append(count)
            remaining_plats -= count
        
        # 最后一个游戏分配剩余的环境
        self.game_allocation.append(remaining_plats)
        
        print(f"进程 {self.process_id}: 游戏分配: {dict(zip(self.available_games, self.game_allocation))}")
    
    def _initialize_environments(self):
        """初始化所有环境"""
        self.envs = []
        self.current_games = []
        self.current_obs = []
        self.episode_rewards = []
        self.episode_steps = []
        
        env_idx = 0
        for game_idx, (game, count) in enumerate(zip(self.available_games, self.game_allocation)):
            for _ in range(count):
                # 多进程环境下不渲染
                current_render_mode = None
                
                try:
                    env = gym.make(game, render_mode=current_render_mode)
                    obs, _ = env.reset()
                    
                    self.envs.append(env)
                    self.current_games.append(game)
                    self.current_obs.append(obs)
                    self.episode_rewards.append(0)
                    self.episode_steps.append(0)
                    
                    print(f"进程 {self.process_id} 环境 {env_idx+1}: {game} 创建成功")
                    env_idx += 1
                    
                except Exception as e:
                    print(f"进程 {self.process_id} 创建环境失败 ({game}): {e}")
        
        if not self.envs:
            raise RuntimeError(f"进程 {self.process_id} 未能成功创建任何环境")
        
        self.actual_num_plats = len(self.envs)
        print(f"进程 {self.process_id} 成功创建 {self.actual_num_plats} 个环境")
    
    def _sample_actions(self):
        """为所有环境采样随机动作"""
        return [env.action_space.sample() for env in self.envs]
    
    def _reset_environment(self, env_idx: int):
        """重置指定环境"""
        obs, _ = self.envs[env_idx].reset()
        self.current_obs[env_idx] = obs
        self.episode_rewards[env_idx] = 0
        self.episode_steps[env_idx] = 0
    
    def generate_data(self):
        """
        生成回放数据
        
        返回:
            str: 保存的数据文件路径
        """
        print(f"进程 {self.process_id} 开始生成数据: {self.actual_num_plats} 个环境 × {self.num_episodes_every_plat} 回合")
        
        # 记录每个环境完成的回合数
        episodes_completed = [0] * self.actual_num_plats
        total_episodes_target = self.actual_num_plats * self.num_episodes_every_plat
        total_episodes_completed = 0
        
        current_episode_data = [[] for _ in range(self.actual_num_plats)]  # 当前回合的数据
        
        step_count = 0
        while total_episodes_completed < total_episodes_target:
            # 采样动作
            actions = self._sample_actions()
            
            # 执行步骤
            for i in range(self.actual_num_plats):
                if episodes_completed[i] >= self.num_episodes_every_plat:
                    continue  # 该环境已完成所需回合数
                
                env = self.envs[i]
                action = actions[i]
                obs1 = self.current_obs[i]
                
                # 执行动作
                obs2, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # 记录步骤数据
                step_data = {
                    'obs': obs1,
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'game': self.current_games[i],
                    'env_id': i,
                    'step': self.episode_steps[i]
                }
                current_episode_data[i].append(step_data)
                
                # 更新状态
                self.current_obs[i] = obs2
                self.episode_rewards[i] += reward
                self.episode_steps[i] += 1
                
                if done:
                    # 回合结束，保存数据
                    episode_info = {
                        'episode_data': current_episode_data[i],
                        'total_reward': self.episode_rewards[i],
                        'total_steps': self.episode_steps[i],
                        'game': self.current_games[i],
                        'env_id': i,
                        'episode_id': episodes_completed[i],
                        'process_id': self.process_id
                    }
                    self.replay_data.append(episode_info)
                    
                    episodes_completed[i] += 1
                    total_episodes_completed += 1
                    
                    print(f"进程 {self.process_id} 环境 {i+1} ({self.current_games[i]}) 完成第 {episodes_completed[i]} 回合, "
                          f"奖励: {self.episode_rewards[i]:.2f}, 步数: {self.episode_steps[i]}, "
                          f"总进度: {total_episodes_completed}/{total_episodes_target}")
                    
                    # 重置环境和数据
                    current_episode_data[i] = []
                    self._reset_environment(i)
            
            step_count += 1
            if step_count % 1000 == 0:
                print(f"进程 {self.process_id} 已执行 {step_count} 步, 完成 {total_episodes_completed}/{total_episodes_target} 回合")
        
        # 保存数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"atari_replay_data_process_{self.process_id}_{timestamp}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'replay_data': self.replay_data,
                'metadata': {
                    'num_plats': self.actual_num_plats,
                    'num_episodes_every_plat': self.num_episodes_every_plat,
                    'games': self.available_games,
                    'game_allocation': self.game_allocation,
                    'total_episodes': len(self.replay_data),
                    'timestamp': timestamp,
                    'process_id': self.process_id
                }
            }, f)
        
        print(f"进程 {self.process_id} 数据已保存到: {filepath}")
        print(f"进程 {self.process_id} 总计生成 {len(self.replay_data)} 个回合的数据")
        
        return filepath
    
    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            if env is not None:
                env.close()
        print(f"进程 {self.process_id} 所有环境已关闭")


class AtariDataProcessor:
    """
    用于处理 Atari 回放数据的类
    """
    
    def __init__(self, window_size: int = 160, stride: int = 80):
        """
        初始化数据处理器
        
        参数:
            window_size (int): 滑动窗口大小
            stride (int): 滑动步长
        """
        self.window_size = window_size
        self.stride = stride
        self.processed_data = []
    
    def load_data(self, filepath: str):
        """
        加载回放数据
        
        参数:
            filepath (str): 数据文件路径
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.replay_data = data['replay_data']
        self.metadata = data['metadata']
        
        print(f"已加载数据: {len(self.replay_data)} 个回合")
        print(f"元数据: {self.metadata}")
    
    def load_multiple_data_files(self, filepaths: List[str]):
        """
        加载多个回放数据文件并合并
        
        参数:
            filepaths (List[str]): 数据文件路径列表
        """
        all_replay_data = []
        all_metadata = []
        
        for filepath in filepaths:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            all_replay_data.extend(data['replay_data'])
            all_metadata.append(data['metadata'])
        
        self.replay_data = all_replay_data
        self.metadata = {
            'total_episodes': len(all_replay_data),
            'source_files': filepaths,
            'process_metadata': all_metadata
        }
        
        print(f"已加载并合并 {len(filepaths)} 个文件的数据: 总计 {len(all_replay_data)} 个回合")
    
    def _extract_episode_steps(self, episode_data: List[dict]) -> List[Tuple]:
        """
        从回合数据中提取步骤信息
        
        参数:
            episode_data (List[dict]): 回合数据
            
        返回:
            List[Tuple]: (obs, action, reward, done) 元组列表
        """
        steps = []
        for step_data in episode_data:
            steps.append((
                step_data['obs'],
                step_data['action'], 
                step_data['reward'],
                step_data['done']
            ))
        return steps
    
    def _create_sliding_windows(self, steps: List[Tuple]) -> List[List[Tuple]]:
        """
        使用滑动窗口创建数据片段
        
        参数:
            steps (List[Tuple]): 步骤数据列表
            
        返回:
            List[List[Tuple]]: 窗口数据列表
        """
        windows = []
        
        # 找到所有游戏结束的位置
        done_positions = [i for i, (_, _, _, done) in enumerate(steps) if done]
        
        start_pos = 0
        while start_pos + self.window_size <= len(steps):
            window = steps[start_pos:start_pos + self.window_size]
            windows.append(window)
            start_pos += self.stride
        
        # 处理最后一个不完整的窗口（如果需要）
        if start_pos < len(steps):
            remaining_steps = len(steps) - start_pos
            if remaining_steps > 0:
                # 向前找补不足的步数
                needed_steps = self.window_size - remaining_steps
                if start_pos >= needed_steps:
                    # 可以向前补充
                    window = steps[start_pos - needed_steps:start_pos + remaining_steps]
                    if len(window) == self.window_size:
                        windows.append(window)
                else:
                    # 不能完全向前补充，从开头开始
                    if len(steps) >= self.window_size:
                        window = steps[-self.window_size:]
                        windows.append(window)
        
        return windows
    
    def process_data(self, save_path: Optional[str] = None) -> List[List[Tuple]]:
        """
        处理所有回放数据
        
        参数:
            save_path (str, optional): 保存处理后数据的路径
            
        返回:
            List[List[Tuple]]: 处理后的窗口数据列表
        """
        if not hasattr(self, 'replay_data'):
            raise ValueError("请先使用 load_data() 加载数据")
        
        print(f"开始处理数据: 窗口大小={self.window_size}, 步长={self.stride}")
        
        all_windows = []
        
        for episode_idx, episode_info in enumerate(self.replay_data):
            episode_data = episode_info['episode_data']
            game = episode_info['game']
            
            # 提取步骤信息
            steps = self._extract_episode_steps(episode_data)
            
            if len(steps) < self.window_size:
                print(f"警告: 回合 {episode_idx} ({game}) 只有 {len(steps)} 步，少于窗口大小 {self.window_size}")
                # 对于步数不足的回合，重复最后的步骤来填充
                if len(steps) > 0:
                    while len(steps) < self.window_size:
                        steps.append(steps[-1])
            
            # 创建滑动窗口
            windows = self._create_sliding_windows(steps)
            
            # 为每个窗口添加元信息
            for window in windows:
                window_info = {
                    'data': window,
                    'game': game,
                    'episode_id': episode_idx,
                    'window_size': len(window)
                }
                all_windows.append(window_info)
            
            if episode_idx % 10 == 0:
                print(f"已处理 {episode_idx + 1}/{len(self.replay_data)} 个回合")
        
        self.processed_data = all_windows
        
        print(f"数据处理完成: 从 {len(self.replay_data)} 个回合生成了 {len(all_windows)} 个数据窗口")
        
        # 保存处理后的数据
        if save_path:
            self.save_processed_data(save_path)
        
        return [window_info['data'] for window_info in all_windows]
    
    def save_processed_data(self, filepath: str):
        """
        保存处理后的数据
        
        参数:
            filepath (str): 保存路径
        """
        if not self.processed_data:
            raise ValueError("没有处理后的数据可保存")
        
        save_data = {
            'processed_data': self.processed_data,
            'processing_params': {
                'window_size': self.window_size,
                'stride': self.stride
            },
            'original_metadata': self.metadata if hasattr(self, 'metadata') else None,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"处理后的数据已保存到: {filepath}")
    
    def get_statistics(self):
        """获取数据统计信息"""
        if not hasattr(self, 'replay_data'):
            return "请先加载数据"
        
        stats = {
            'total_episodes': len(self.replay_data),
            'games': {},
            'total_steps': 0,
            'total_reward': 0
        }
        
        for episode_info in self.replay_data:
            game = episode_info['game']
            if game not in stats['games']:
                stats['games'][game] = {'episodes': 0, 'steps': 0, 'reward': 0}
            
            stats['games'][game]['episodes'] += 1
            stats['games'][game]['steps'] += episode_info['total_steps'] 
            stats['games'][game]['reward'] += episode_info['total_reward']
            stats['total_steps'] += episode_info['total_steps']
            stats['total_reward'] += episode_info['total_reward']
        
        if hasattr(self, 'processed_data') and self.processed_data:
            stats['processed_windows'] = len(self.processed_data)
        
        return stats


def generate_data_worker(args):
    """
    多进程工作函数
    
    参数:
        args: 包含所有必要参数的元组
    """
    (process_id, num_plats_per_process, num_episodes_every_plat, 
     game_ratios, render_mode, save_dir, games, seed) = args
    
    # 设置每个进程的随机种子
    if seed is not None:
        np.random.seed(seed + process_id)
    
    try:
        generator = AtariDataGenerator(
            num_plats=num_plats_per_process,
            num_episodes_every_plat=num_episodes_every_plat,
            game_ratios=game_ratios,
            render_mode=None,  # 多进程环境下不渲染
            save_dir=save_dir,
            custom_games=games,
            process_id=process_id
        )
        
        data_file = generator.generate_data()
        generator.close()
        
        return data_file
        
    except Exception as e:
        print(f"进程 {process_id} 生成数据失败: {e}")
        return None


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Atari游戏数据生成和处理工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 主要操作模式
    parser.add_argument('--mode', type=str, choices=['generate', 'process', 'both'], 
                       default='both', help='运行模式: generate(仅生成), process(仅处理), both(生成并处理)')
    
    # ========== 数据生成参数 ==========
    generate_group = parser.add_argument_group('数据生成参数')
    generate_group.add_argument('--num-plats', type=int, default=1,
                               help='并行环境数量')
    generate_group.add_argument('--num-episodes-every-plat', type=int, default=1,
                               help='每个环境运行的游戏场数')
    generate_group.add_argument('--game-ratios', type=str, default=None,
                               help='游戏比例，用逗号分隔的浮点数，如 "0.4,0.4,0.2"。如果不指定则平均分配')
    generate_group.add_argument('--games', type=str, default=None,
                               help='要使用的游戏列表，用逗号分隔，如 "Seaquest-v4,Riverraid-v4"。如果不指定则使用默认游戏')
    generate_group.add_argument('--render-mode', type=str, choices=['human', 'rgb_array', None], 
                               default='rgb_array', help='渲染模式')
    generate_group.add_argument('--save-dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data/replay_data'),
                               help='原始回放数据保存目录（代码文件的上上级目录下）')
    generate_group.add_argument('--num-processes', type=int, default=4,
                               help='并行进程数量')
    
    # ========== 数据处理参数 ==========
    process_group = parser.add_argument_group('数据处理参数')
    process_group.add_argument('--window-size', type=int, default=512,
                              help='滑动窗口大小')
    process_group.add_argument('--stride', type=int, default=256,
                              help='滑动窗口步长')
    process_group.add_argument('--input-file', type=str, default=None,
                              help='要处理的数据文件路径（仅在process模式下需要）')
    process_group.add_argument('--output-file', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data/processed_data'),
                              help='处理后数据的保存路径（可选）')
    
    # ========== 其他参数 ==========
    other_group = parser.add_argument_group('其他参数')
    other_group.add_argument('--seed', type=int, default=None,
                            help='随机种子')
    other_group.add_argument('--verbose', action='store_true',
                            help='显示详细输出')
    other_group.add_argument('--config', type=str, default=None,
                            help='配置文件路径（JSON格式），可以覆盖命令行参数')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """从JSON文件加载配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config_template(filepath: str):
    """保存配置文件模板"""
    template = {
        "mode": "both",
        "num_plats": 1,
        "num_episodes_every_plat": 1,
        "game_ratios": [0.33, 0.33, 0.34],
        "games": ["Seaquest-v4", "Riverraid-v4", "ChopperCommand-v4"],
        "render_mode": "rgb_array",
        "save_dir": "../replay_data",
        "output_file": "../processed_data",
        "window_size": 160,
        "stride": 80,
        "seed": 42,
        "verbose": True,
        "num_processes": 80
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    print(f"配置文件模板已保存到: {filepath}")


def merge_config_and_args(args, config: dict = None) -> dict:
    """合并配置文件和命令行参数"""
    # 从args转换为字典
    params = vars(args).copy()
    
    # 如果有配置文件，优先使用配置文件的值
    if config:
        for key, value in config.items():
            # 将配置文件中的key转换为args格式
            key_mapped = key.replace('_', '_')  # 保持一致性
            if key_mapped in params:
                # 只有当命令行参数是默认值时才使用配置文件值
                if key == 'games' and value and params.get('games') is None:
                    params['games'] = ','.join(value)
                elif key == 'game_ratios' and value and params.get('game_ratios') is None:
                    params['game_ratios'] = ','.join(map(str, value))
                elif params.get(key_mapped) == argparse.Namespace.__dict__.get(key_mapped):
                    params[key_mapped] = value
    
    return params


def parse_game_ratios(ratios_str: str) -> List[float]:
    """解析游戏比例字符串"""
    if not ratios_str:
        return None
    
    try:
        ratios = [float(x.strip()) for x in ratios_str.split(',')]
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"游戏比例总和必须为1，当前为: {sum(ratios)}")
        return ratios
    except ValueError as e:
        raise ValueError(f"解析游戏比例失败: {e}")


def parse_games(games_str: str) -> List[str]:
    """解析游戏列表字符串"""
    if not games_str:
        return None
    
    games = [game.strip() for game in games_str.split(',')]
    return games


def main():
    """主函数"""
    args = parse_args()
    
    # 处理配置文件
    config = None
    if args.config:
        if args.config == 'template':
            # 生成配置文件模板
            save_config_template('config_template.json')
            return
        else:
            config = load_config(args.config)
    
    # 合并配置，最终配置信息存储在params中
    params = merge_config_and_args(args, config)

    # 只初始化一个日志文件
    log_mode = params['mode']
    os.makedirs(params["save_dir"], exist_ok=True)
    log_filename = os.path.join(params["save_dir"], f"log_{log_mode}.txt")
    counter = 0
    while os.path.exists(log_filename):
        counter += 1
        log_filename = os.path.join(params["save_dir"], f"log_{log_mode}_{counter}.txt")

    logging.basicConfig(filename=log_filename,
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        filemode='w')

    logging.info("程序启动")
    logging.info(f"选择的模式: {params['mode']}")
    logging.info(f"并行进程数: {params['num_processes']}")
    logging.info(f"基本游戏信息: {params['games'] if params['games'] else '使用默认游戏列表'}")
    logging.info(f"最终配置: {params}")
    logging.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"配置文件信息: {args.config if args.config else '未提供配置文件'}")
    
    # 设置随机种子
    if params['seed'] is not None:
        np.random.seed(params['seed'])
        print(f"设置随机种子: {params['seed']}")
    
    # 解析游戏相关参数
    games = parse_games(params['games']) if params['games'] else None
    game_ratios = parse_game_ratios(params['game_ratios']) if params['game_ratios'] else None
    
    if params['verbose']:
        print("运行参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print()
    
    data_files = []
    
    # 1. 数据生成
    if params['mode'] in ['generate', 'both']:
        print("=" * 50)
        print("开始并行生成 Atari 游戏数据")
        print("=" * 50)
        
        num_processes = params['num_processes']
        
        # 计算每个进程分配的环境数量
        num_plats_per_process = max(1, params['num_plats'] // num_processes)
        remaining_plats = params['num_plats'] % num_processes
        
        print(f"总进程数: {num_processes}")
        print(f"每个进程环境数: {num_plats_per_process}")
        print(f"剩余环境数: {remaining_plats}")
        
        # 准备多进程参数
        process_args = []
        for process_id in range(num_processes):
            # 前 remaining_plats 个进程多分配一个环境
            current_num_plats = num_plats_per_process + (1 if process_id < remaining_plats else 0)
            
            if current_num_plats > 0:  # 只有分配到环境的进程才参与
                process_args.append((
                    process_id,
                    current_num_plats,
                    params['num_episodes_every_plat'],
                    game_ratios,
                    params['render_mode'],
                    params['save_dir'],
                    games,
                    params['seed']
                ))
        
        print(f"实际使用进程数: {len(process_args)}")
        
        # 使用多进程生成数据
        start_time = time.time()
        
        try:
            with Pool(processes=len(process_args)) as pool:
                results = pool.map(generate_data_worker, process_args)
            
            # 过滤成功的结果
            data_files = [result for result in results if result is not None]
            
            end_time = time.time()
            
            print(f"并行数据生成完成!")
            print(f"成功生成 {len(data_files)} 个数据文件")
            print(f"总耗时: {end_time - start_time:.2f} 秒")
            
            for i, file in enumerate(data_files):
                print(f"  文件 {i+1}: {file}")
                
        except KeyboardInterrupt:
            print("\n数据生成被中断")
            return
        except Exception as e:
            print(f"并行数据生成失败: {e}")
            return
    
    # 2. 数据处理
    if params['mode'] in ['process', 'both']:
        print("\n" + "=" * 50)
        print("开始处理数据")
        print("=" * 50)
        
        # 确定输入文件
        if params['mode'] == 'process':
            if not params['input_file']:
                print("错误: process模式需要指定--input-file参数")
                return
            
            # 支持单个文件或文件夹
            input_path = params['input_file']
            if os.path.isdir(input_path):
                # 如果是文件夹，找到所有.pkl文件
                input_files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                              if f.endswith('.pkl') and 'atari_replay_data' in f]
            else:
                input_files = [input_path]
        else:
            input_files = data_files
        
        if not input_files:
            print("错误: 没有找到要处理的数据文件")
            return
        
        # 检查文件是否存在
        valid_files = [f for f in input_files if os.path.exists(f)]
        if not valid_files:
            print("错误: 输入文件都不存在")
            return
        
        print(f"将处理 {len(valid_files)} 个数据文件")
        
        processor = AtariDataProcessor(
            window_size=params['window_size'], 
            stride=params['stride']
        )
        
        try:
            # 加载多个数据文件
            if len(valid_files) == 1:
                processor.load_data(valid_files[0])
            else:
                processor.load_multiple_data_files(valid_files)
            
            # 显示统计信息
            if params['verbose']:
                stats = processor.get_statistics()
                print(f"数据统计: {stats}")
            
            # 确定输出文件
            if params['output_file']:
                output_file = params['output_file']
                if not output_file.endswith('.pkl'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = os.path.join(output_file, f"processed_data_{timestamp}.pkl")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(params['save_dir'], f"processed_data_{timestamp}.pkl")
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 处理数据
            windows = processor.process_data(save_path=output_file)
            
            # 显示处理结果
            if params['verbose']:
                final_stats = processor.get_statistics()
                print(f"最终统计: {final_stats}")
            
            print(f"数据处理完成: {output_file}")
            
        except Exception as e:
            print(f"数据处理失败: {e}")
            return
    
    print(f"\n任务完成!")


# 示例使用和主程序入口
if __name__ == '__main__':
    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)
    main()# ReplayBuffer.py

gym.register_envs(ale_py)  # 注册 Atari 环境


class AtariDataGenerator:
    """
    用于生成 Atari 游戏回放数据的类
    """
    
    def __init__(self, 
                 num_plats: int = 5,
                 num_episodes_every_plat: int = 10,
                 game_ratios: Optional[List[float]] = None,
                 render_mode: Optional[str] = None,
                 save_dir: str = "replay_data",
                 custom_games: Optional[List[str]] = None,
                 process_id: int = 0):
        """
        初始化数据生成器
        
        参数:
            num_plats (int): 并行环境数量
            num_episodes_every_plat (int): 每个环境运行的回合数
            game_ratios (List[float], optional): 各游戏类型的比例，默认平均分配
            render_mode (str, optional): 渲染模式
            save_dir (str): 数据保存目录
            custom_games (List[str], optional): 自定义游戏列表
            process_id (int): 进程ID，用于区分不同进程
        """
        self.num_plats = num_plats
        self.num_episodes_every_plat = num_episodes_every_plat
        self.save_dir = save_dir
        self.render_mode = render_mode
        self.process_id = process_id
        
        # 设置可用游戏列表
        if custom_games:
            self.available_games = custom_games
        else:
            # 默认游戏列表（可扩展）
            self.available_games = [
                "Seaquest-v4",
                "Riverraid-v4", 
                "ChopperCommand-v4"
            ]
        
        print(f"进程 {self.process_id}: 使用游戏列表: {self.available_games}")
        
        # 设置游戏比例
        if game_ratios is None:
            self.game_ratios = [1.0 / len(self.available_games)] * len(self.available_games)
        else:
            assert len(game_ratios) == len(self.available_games), f"游戏比例数量({len(game_ratios)})必须与游戏数量({len(self.available_games)})一致"
            assert abs(sum(game_ratios) - 1.0) < 1e-6, "游戏比例总和必须为1"
            self.game_ratios = game_ratios
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 计算每种游戏的环境数量
        self._allocate_games()
        
        # 初始化环境
        self._initialize_environments()
        
        # 存储所有回放数据
        self.replay_data = []
        
    def _allocate_games(self):
        """根据比例分配各游戏的环境数量"""
        self.game_allocation = []
        remaining_plats = self.num_plats
        
        for i, ratio in enumerate(self.game_ratios[:-1]):
            count = int(self.num_plats * ratio)
            self.game_allocation.append(count)
            remaining_plats -= count
        
        # 最后一个游戏分配剩余的环境
        self.game_allocation.append(remaining_plats)
        
        print(f"进程 {self.process_id}: 游戏分配: {dict(zip(self.available_games, self.game_allocation))}")
    
    def _initialize_environments(self):
        """初始化所有环境"""
        self.envs = []
        self.current_games = []
        self.current_obs = []
        self.episode_rewards = []
        self.episode_steps = []
        
        env_idx = 0
        for game_idx, (game, count) in enumerate(zip(self.available_games, self.game_allocation)):
            for _ in range(count):
                # 多进程环境下不渲染
                current_render_mode = None
                
                try:
                    env = gym.make(game, render_mode=current_render_mode)
                    obs, _ = env.reset()
                    
                    self.envs.append(env)
                    self.current_games.append(game)
                    self.current_obs.append(obs)
                    self.episode_rewards.append(0)
                    self.episode_steps.append(0)
                    
                    print(f"进程 {self.process_id} 环境 {env_idx+1}: {game} 创建成功")
                    env_idx += 1
                    
                except Exception as e:
                    print(f"进程 {self.process_id} 创建环境失败 ({game}): {e}")
        
        if not self.envs:
            raise RuntimeError(f"进程 {self.process_id} 未能成功创建任何环境")
        
        self.actual_num_plats = len(self.envs)
        print(f"进程 {self.process_id} 成功创建 {self.actual_num_plats} 个环境")
    
    def _sample_actions(self):
        """为所有环境采样随机动作"""
        return [env.action_space.sample() for env in self.envs]
    
    def _reset_environment(self, env_idx: int):
        """重置指定环境"""
        obs, _ = self.envs[env_idx].reset()
        self.current_obs[env_idx] = obs
        self.episode_rewards[env_idx] = 0
        self.episode_steps[env_idx] = 0
    
    def generate_data(self):
        """
        生成回放数据
        
        返回:
            str: 保存的数据文件路径
        """
        print(f"进程 {self.process_id} 开始生成数据: {self.actual_num_plats} 个环境 × {self.num_episodes_every_plat} 回合")
        
        # 记录每个环境完成的回合数
        episodes_completed = [0] * self.actual_num_plats
        total_episodes_target = self.actual_num_plats * self.num_episodes_every_plat
        total_episodes_completed = 0
        
        current_episode_data = [[] for _ in range(self.actual_num_plats)]  # 当前回合的数据
        
        step_count = 0
        while total_episodes_completed < total_episodes_target:
            # 采样动作
            actions = self._sample_actions()
            
            # 执行步骤
            for i in range(self.actual_num_plats):
                if episodes_completed[i] >= self.num_episodes_every_plat:
                    continue  # 该环境已完成所需回合数
                
                env = self.envs[i]
                action = actions[i]
                obs1 = self.current_obs[i]
                
                # 执行动作
                obs2, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # 记录步骤数据
                step_data = {
                    'obs': obs1,
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'game': self.current_games[i],
                    'env_id': i,
                    'step': self.episode_steps[i]
                }
                current_episode_data[i].append(step_data)
                
                # 更新状态
                self.current_obs[i] = obs2
                self.episode_rewards[i] += reward
                self.episode_steps[i] += 1
                
                if done:
                    # 回合结束，保存数据
                    episode_info = {
                        'episode_data': current_episode_data[i],
                        'total_reward': self.episode_rewards[i],
                        'total_steps': self.episode_steps[i],
                        'game': self.current_games[i],
                        'env_id': i,
                        'episode_id': episodes_completed[i],
                        'process_id': self.process_id
                    }
                    self.replay_data.append(episode_info)
                    
                    episodes_completed[i] += 1
                    total_episodes_completed += 1
                    
                    print(f"进程 {self.process_id} 环境 {i+1} ({self.current_games[i]}) 完成第 {episodes_completed[i]} 回合, "
                          f"奖励: {self.episode_rewards[i]:.2f}, 步数: {self.episode_steps[i]}, "
                          f"总进度: {total_episodes_completed}/{total_episodes_target}")
                    
                    # 重置环境和数据
                    current_episode_data[i] = []
                    self._reset_environment(i)
            
            step_count += 1
            if step_count % 1000 == 1:
                print(f"进程 {self.process_id} 已执行 {step_count} 步, 完成 {total_episodes_completed}/{total_episodes_target} 回合")
        
        # 保存数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"atari_replay_data_process_{self.process_id}_{timestamp}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'replay_data': self.replay_data,
                'metadata': {
                    'num_plats': self.actual_num_plats,
                    'num_episodes_every_plat': self.num_episodes_every_plat,
                    'games': self.available_games,
                    'game_allocation': self.game_allocation,
                    'total_episodes': len(self.replay_data),
                    'timestamp': timestamp,
                    'process_id': self.process_id
                }
            }, f)
        
        print(f"进程 {self.process_id} 数据已保存到: {filepath}")
        print(f"进程 {self.process_id} 总计生成 {len(self.replay_data)} 个回合的数据")
        
        return filepath
    
    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            if env is not None:
                env.close()
        print(f"进程 {self.process_id} 所有环境已关闭")


class AtariDataProcessor:
    """
    用于处理 Atari 回放数据的类
    """
    
    def __init__(self, window_size: int = 160, stride: int = 80):
        """
        初始化数据处理器
        
        参数:
            window_size (int): 滑动窗口大小
            stride (int): 滑动步长
        """
        self.window_size = window_size
        self.stride = stride
        self.processed_data = []
    
    def load_data(self, filepath: str):
        """
        加载回放数据
        
        参数:
            filepath (str): 数据文件路径
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.replay_data = data['replay_data']
        self.metadata = data['metadata']
        
        print(f"已加载数据: {len(self.replay_data)} 个回合")
        print(f"元数据: {self.metadata}")
    
    def load_multiple_data_files(self, filepaths: List[str]):
        """
        加载多个回放数据文件并合并
        
        参数:
            filepaths (List[str]): 数据文件路径列表
        """
        all_replay_data = []
        all_metadata = []
        
        for filepath in filepaths:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            all_replay_data.extend(data['replay_data'])
            all_metadata.append(data['metadata'])
        
        self.replay_data = all_replay_data
        self.metadata = {
            'total_episodes': len(all_replay_data),
            'source_files': filepaths,
            'process_metadata': all_metadata
        }
        
        print(f"已加载并合并 {len(filepaths)} 个文件的数据: 总计 {len(all_replay_data)} 个回合")
    
    def _extract_episode_steps(self, episode_data: List[dict]) -> List[Tuple]:
        """
        从回合数据中提取步骤信息
        
        参数:
            episode_data (List[dict]): 回合数据
            
        返回:
            List[Tuple]: (obs, action, reward, done) 元组列表
        """
        steps = []
        for step_data in episode_data:
            steps.append((
                step_data['obs'],
                step_data['action'], 
                step_data['reward'],
                step_data['done']
            ))
        return steps
    
    def _create_sliding_windows(self, steps: List[Tuple]) -> List[List[Tuple]]:
        """
        使用滑动窗口创建数据片段
        
        参数:
            steps (List[Tuple]): 步骤数据列表
            
        返回:
            List[List[Tuple]]: 窗口数据列表
        """
        windows = []
        
        # 找到所有游戏结束的位置
        done_positions = [i for i, (_, _, _, done) in enumerate(steps) if done]
        
        start_pos = 0
        while start_pos + self.window_size <= len(steps):
            window = steps[start_pos:start_pos + self.window_size]
            windows.append(window)
            start_pos += self.stride
        
        # 处理最后一个不完整的窗口（如果需要）
        if start_pos < len(steps):
            remaining_steps = len(steps) - start_pos
            if remaining_steps > 0:
                # 向前找补不足的步数
                needed_steps = self.window_size - remaining_steps
                if start_pos >= needed_steps:
                    # 可以向前补充
                    window = steps[start_pos - needed_steps:start_pos + remaining_steps]
                    if len(window) == self.window_size:
                        windows.append(window)
                else:
                    # 不能完全向前补充，从开头开始
                    if len(steps) >= self.window_size:
                        window = steps[-self.window_size:]
                        windows.append(window)
        
        return windows
    
    def process_data(self, save_path: Optional[str] = None) -> List[List[Tuple]]:
        """
        处理所有回放数据
        
        参数:
            save_path (str, optional): 保存处理后数据的路径
            
        返回:
            List[List[Tuple]]: 处理后的窗口数据列表
        """
        if not hasattr(self, 'replay_data'):
            raise ValueError("请先使用 load_data() 加载数据")
        
        print(f"开始处理数据: 窗口大小={self.window_size}, 步长={self.stride}")
        
        all_windows = []
        
        for episode_idx, episode_info in enumerate(self.replay_data):
            episode_data = episode_info['episode_data']
            game = episode_info['game']
            
            # 提取步骤信息
            steps = self._extract_episode_steps(episode_data)
            
            if len(steps) < self.window_size:
                print(f"警告: 回合 {episode_idx} ({game}) 只有 {len(steps)} 步，少于窗口大小 {self.window_size}")
                # 对于步数不足的回合，重复最后的步骤来填充
                if len(steps) > 0:
                    while len(steps) < self.window_size:
                        steps.append(steps[-1])
            
            # 创建滑动窗口
            windows = self._create_sliding_windows(steps)
            
            # 为每个窗口添加元信息
            for window in windows:
                window_info = {
                    'data': window,
                    'game': game,
                    'episode_id': episode_idx,
                    'window_size': len(window)
                }
                all_windows.append(window_info)
            
            if episode_idx % 10 == 0:
                print(f"已处理 {episode_idx + 1}/{len(self.replay_data)} 个回合")
        
        self.processed_data = all_windows
        
        print(f"数据处理完成: 从 {len(self.replay_data)} 个回合生成了 {len(all_windows)} 个数据窗口")
        
        # 保存处理后的数据
        if save_path:
            self.save_processed_data(save_path)
        
        return [window_info['data'] for window_info in all_windows]
    
    def save_processed_data(self, filepath: str):
        """
        保存处理后的数据
        
        参数:
            filepath (str): 保存路径
        """
        if not self.processed_data:
            raise ValueError("没有处理后的数据可保存")
        
        save_data = {
            'processed_data': self.processed_data,
            'processing_params': {
                'window_size': self.window_size,
                'stride': self.stride
            },
            'original_metadata': self.metadata if hasattr(self, 'metadata') else None,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"处理后的数据已保存到: {filepath}")
    
    def get_statistics(self):
        """获取数据统计信息"""
        if not hasattr(self, 'replay_data'):
            return "请先加载数据"
        
        stats = {
            'total_episodes': len(self.replay_data),
            'games': {},
            'total_steps': 0,
            'total_reward': 0
        }
        
        for episode_info in self.replay_data:
            game = episode_info['game']
            if game not in stats['games']:
                stats['games'][game] = {'episodes': 0, 'steps': 0, 'reward': 0}
            
            stats['games'][game]['episodes'] += 1
            stats['games'][game]['steps'] += episode_info['total_steps'] 
            stats['games'][game]['reward'] += episode_info['total_reward']
            stats['total_steps'] += episode_info['total_steps']
            stats['total_reward'] += episode_info['total_reward']
        
        if hasattr(self, 'processed_data') and self.processed_data:
            stats['processed_windows'] = len(self.processed_data)
        
        return stats


def generate_data_worker(args):
    """
    多进程工作函数
    
    参数:
        args: 包含所有必要参数的元组
    """
    (process_id, num_plats_per_process, num_episodes_every_plat, 
     game_ratios, render_mode, save_dir, games, seed) = args
    
    # 设置每个进程的随机种子
    if seed is not None:
        np.random.seed(seed + process_id)
    
    try:
        generator = AtariDataGenerator(
            num_plats=num_plats_per_process,
            num_episodes_every_plat=num_episodes_every_plat,
            game_ratios=game_ratios,
            render_mode=None,  # 多进程环境下不渲染
            save_dir=save_dir,
            custom_games=games,
            process_id=process_id
        )
        
        data_file = generator.generate_data()
        generator.close()
        
        return data_file
        
    except Exception as e:
        print(f"进程 {process_id} 生成数据失败: {e}")
        return None


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Atari游戏数据生成和处理工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 主要操作模式
    parser.add_argument('--mode', type=str, choices=['generate', 'process', 'both'], 
                       default='both', help='运行模式: generate(仅生成), process(仅处理), both(生成并处理)')
    
    # ========== 数据生成参数 ==========
    generate_group = parser.add_argument_group('数据生成参数')
    generate_group.add_argument('--num-plats', type=int, default=1,
                               help='并行环境数量')
    generate_group.add_argument('--num-episodes-every-plat', type=int, default=1,
                               help='每个环境运行的游戏场数')
    generate_group.add_argument('--game-ratios', type=str, default=None,
                               help='游戏比例，用逗号分隔的浮点数，如 "0.4,0.4,0.2"。如果不指定则平均分配')
    generate_group.add_argument('--games', type=str, default=None,
                               help='要使用的游戏列表，用逗号分隔，如 "Seaquest-v4,Riverraid-v4"。如果不指定则使用默认游戏')
    generate_group.add_argument('--render-mode', type=str, choices=['human', 'rgb_array', None], 
                               default='rgb_array', help='渲染模式')
    generate_group.add_argument('--save-dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data/replay_data'),
                               help='原始回放数据保存目录（代码文件的上上级目录下）')
    generate_group.add_argument('--num-processes', type=int, default=4,
                               help='并行进程数量')
    
    # ========== 数据处理参数 ==========
    process_group = parser.add_argument_group('数据处理参数')
    process_group.add_argument('--window-size', type=int, default=512,
                              help='滑动窗口大小')
    process_group.add_argument('--stride', type=int, default=256,
                              help='滑动窗口步长')
    process_group.add_argument('--input-file', type=str, default=None,
                              help='要处理的数据文件路径（仅在process模式下需要）')
    process_group.add_argument('--output-file', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data/processed_data'),
                              help='处理后数据的保存路径（可选）')
    
    # ========== 其他参数 ==========
    other_group = parser.add_argument_group('其他参数')
    other_group.add_argument('--seed', type=int, default=None,
                            help='随机种子')
    other_group.add_argument('--verbose', action='store_true',
                            help='显示详细输出')
    other_group.add_argument('--config', type=str, default=None,
                            help='配置文件路径（JSON格式），可以覆盖命令行参数')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """从JSON文件加载配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config_template(filepath: str):
    """保存配置文件模板"""
    template = {
        "mode": "both",
        "num_plats": 1,
        "num_episodes_every_plat": 1,
        "game_ratios": [0.33, 0.33, 0.34],
        "games": ["Seaquest-v4", "Riverraid-v4", "ChopperCommand-v4"],
        "render_mode": "rgb_array",
        "save_dir": "../replay_data",
        "output_file": "../processed_data",
        "window_size": 160,
        "stride": 80,
        "seed": 42,
        "verbose": True,
        "num_processes": 64
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    print(f"配置文件模板已保存到: {filepath}")


def merge_config_and_args(args, config: dict = None) -> dict:
    """合并配置文件和命令行参数"""
    # 从args转换为字典
    params = vars(args).copy()
    
    # 如果有配置文件，优先使用配置文件的值
    if config:
        for key, value in config.items():
            # 将配置文件中的key转换为args格式
            key_mapped = key.replace('_', '_')  # 保持一致性
            if key_mapped in params:
                # 只有当命令行参数是默认值时才使用配置文件值
                if key == 'games' and value and params.get('games') is None:
                    params['games'] = ','.join(value)
                elif key == 'game_ratios' and value and params.get('game_ratios') is None:
                    params['game_ratios'] = ','.join(map(str, value))
                elif params.get(key_mapped) == argparse.Namespace.__dict__.get(key_mapped):
                    params[key_mapped] = value
    
    return params


def parse_game_ratios(ratios_str: str) -> List[float]:
    """解析游戏比例字符串"""
    if not ratios_str:
        return None
    
    try:
        ratios = [float(x.strip()) for x in ratios_str.split(',')]
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"游戏比例总和必须为1，当前为: {sum(ratios)}")
        return ratios
    except ValueError as e:
        raise ValueError(f"解析游戏比例失败: {e}")


def parse_games(games_str: str) -> List[str]:
    """解析游戏列表字符串"""
    if not games_str:
        return None
    
    games = [game.strip() for game in games_str.split(',')]
    return games


def main():
    """主函数"""
    args = parse_args()
    
    # 处理配置文件
    config = None
    if args.config:
        if args.config == 'template':
            # 生成配置文件模板
            save_config_template('config_template.json')
            return
        else:
            config = load_config(args.config)
    
    # 合并配置，最终配置信息存储在params中
    params = merge_config_and_args(args, config)

    # 只初始化一个日志文件
    log_mode = params['mode']
    os.makedirs(params["save_dir"], exist_ok=True)
    log_filename = os.path.join(params["save_dir"], f"log_{log_mode}.txt")
    counter = 0
    while os.path.exists(log_filename):
        counter += 1
        log_filename = os.path.join(params["save_dir"], f"log_{log_mode}_{counter}.txt")

    logging.basicConfig(filename=log_filename,
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        filemode='w')

    logging.info("程序启动")
    logging.info(f"选择的模式: {params['mode']}")
    logging.info(f"并行进程数: {params['num_processes']}")
    logging.info(f"基本游戏信息: {params['games'] if params['games'] else '使用默认游戏列表'}")
    logging.info(f"最终配置: {params}")
    logging.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"配置文件信息: {args.config if args.config else '未提供配置文件'}")
    
    # 设置随机种子
    if params['seed'] is not None:
        np.random.seed(params['seed'])
        print(f"设置随机种子: {params['seed']}")
    
    # 解析游戏相关参数
    games = parse_games(params['games']) if params['games'] else None
    game_ratios = parse_game_ratios(params['game_ratios']) if params['game_ratios'] else None
    
    if params['verbose']:
        print("运行参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print()
    
    data_files = []
    
    # 1. 数据生成
    if params['mode'] in ['generate', 'both']:
        print("=" * 50)
        print("开始并行生成 Atari 游戏数据")
        print("=" * 50)
        
        num_processes = params['num_processes']
        
        # 计算每个进程分配的环境数量
        num_plats_per_process = max(1, params['num_plats'] // num_processes)
        remaining_plats = params['num_plats'] % num_processes
        
        print(f"总进程数: {num_processes}")
        print(f"每个进程环境数: {num_plats_per_process}")
        print(f"剩余环境数: {remaining_plats}")
        
        # 准备多进程参数
        process_args = []
        for process_id in range(num_processes):
            # 前 remaining_plats 个进程多分配一个环境
            current_num_plats = num_plats_per_process + (1 if process_id < remaining_plats else 0)
            
            if current_num_plats > 0:  # 只有分配到环境的进程才参与
                process_args.append((
                    process_id,
                    current_num_plats,
                    params['num_episodes_every_plat'],
                    game_ratios,
                    params['render_mode'],
                    params['save_dir'],
                    games,
                    params['seed']
                ))
        
        print(f"实际使用进程数: {len(process_args)}")
        
        # 使用多进程生成数据
        start_time = time.time()
        
        try:
            with Pool(processes=len(process_args)) as pool:
                results = pool.map(generate_data_worker, process_args)
            
            # 过滤成功的结果
            data_files = [result for result in results if result is not None]
            
            end_time = time.time()
            
            print(f"并行数据生成完成!")
            print(f"成功生成 {len(data_files)} 个数据文件")
            print(f"总耗时: {end_time - start_time:.2f} 秒")
            
            for i, file in enumerate(data_files):
                print(f"  文件 {i+1}: {file}")
                
        except KeyboardInterrupt:
            print("\n数据生成被中断")
            return
        except Exception as e:
            print(f"并行数据生成失败: {e}")
            return
    
    # 2. 数据处理
    if params['mode'] in ['process', 'both']:
        print("\n" + "=" * 50)
        print("开始处理数据")
        print("=" * 50)
        
        # 确定输入文件
        if params['mode'] == 'process':
            if not params['input_file']:
                print("错误: process模式需要指定--input-file参数")
                return
            
            # 支持单个文件或文件夹
            input_path = params['input_file']
            if os.path.isdir(input_path):
                # 如果是文件夹，找到所有.pkl文件
                input_files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                              if f.endswith('.pkl') and 'atari_replay_data' in f]
            else:
                input_files = [input_path]
        else:
            input_files = data_files
        
        if not input_files:
            print("错误: 没有找到要处理的数据文件")
            return
        
        # 检查文件是否存在
        valid_files = [f for f in input_files if os.path.exists(f)]
        if not valid_files:
            print("错误: 输入文件都不存在")
            return
        
        print(f"将处理 {len(valid_files)} 个数据文件")
        
        processor = AtariDataProcessor(
            window_size=params['window_size'], 
            stride=params['stride']
        )
        
        try:
            # 加载多个数据文件
            if len(valid_files) == 1:
                processor.load_data(valid_files[0])
            else:
                processor.load_multiple_data_files(valid_files)
            
            # 显示统计信息
            if params['verbose']:
                stats = processor.get_statistics()
                print(f"数据统计: {stats}")
            
            # 确定输出文件
            if params['output_file']:
                output_file = params['output_file']
                if not output_file.endswith('.pkl'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = os.path.join(output_file, f"processed_data_{timestamp}.pkl")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(params['save_dir'], f"processed_data_{timestamp}.pkl")
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 处理数据
            windows = processor.process_data(save_path=output_file)
            
            # 显示处理结果
            if params['verbose']:
                final_stats = processor.get_statistics()
                print(f"最终统计: {final_stats}")
            
            print(f"数据处理完成: {output_file}")
            
        except Exception as e:
            print(f"数据处理失败: {e}")
            return
    
    print(f"\n任务完成!")


# 示例使用和主程序入口
if __name__ == '__main__':
    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)
    main()