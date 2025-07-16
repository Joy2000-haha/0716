import gymnasium as gym
import ale_py
import numpy as np
import pickle
import os
import argparse
import json
import signal
import sys
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import logging
import multiprocessing as mp
from multiprocessing import Pool, Manager, Value, Lock
import time
import traceback
from pathlib import Path

gym.register_envs(ale_py)


class InterruptHandler:
    """处理中断信号，确保数据安全保存"""
    
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        # 子进程中，只标记状态，让主循环来处理退出
        self.interrupted = True


class SafeDataSaver:
    """安全的数据保存器，支持中断恢复"""
    
    def __init__(self, save_dir: str, process_id: int):
        self.save_dir = Path(save_dir)
        self.process_id = process_id
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.temp_file = self.save_dir / f"temp_process_{process_id}.pkl"
        self.checkpoint_interval = 10
        
    def save_checkpoint(self, data: List[Dict], episode_count: int):
        """保存检查点"""
        try:
            # 检查点保存的是原始嵌套结构，因为它需要支持恢复
            checkpoint_data = {
                'replay_data': data,
                'episode_count': episode_count,
                'process_id': self.process_id,
                'timestamp': datetime.now().isoformat(),
            }
            with open(self.temp_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        except Exception as e:
            print(f"进程 {self.process_id} 保存检查点失败: {e}")
    
    def load_checkpoint(self) -> Tuple[List[Dict], int]:
        """加载检查点数据"""
        if not self.temp_file.exists():
            return [], 0
            
        try:
            with open(self.temp_file, 'rb') as f:
                data = pickle.load(f)
            return data['replay_data'], data['episode_count']
        except Exception as e:
            print(f"进程 {self.process_id} 加载检查点失败: {e}")
            return [], 0


class AtariEnvironment:
    """单个Atari环境的封装"""
    
    def __init__(self, game_name: str, env_id: int, process_id: int):
        self.game_name = game_name
        self.env_id = env_id
        self.process_id = process_id
        self.env = None
        self.current_obs = None
        self.episode_reward = 0
        self.episode_steps = 0
        self.current_episode_data = []
        
        self._create_environment()
    
    def _create_environment(self):
        try:
            self.env = gym.make(self.game_name, render_mode="rgb_array")
            self.reset()
        except Exception as e:
            print(f"进程 {self.process_id} 创建环境失败 ({self.game_name}): {e}")
            raise
    
    def reset(self):
        if self.env is not None:
            self.current_obs, _ = self.env.reset()
            self.episode_reward = 0
            self.episode_steps = 0
            self.current_episode_data = []
    
    def step(self, action: int) -> Tuple[bool, Dict]:
        if self.env is None:
            return False, {}
        
        obs1 = self.current_obs
        obs2, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        step_data = {
            'obs': obs1,
            'action': action,
            'reward': reward,
            'done': done,
        }
        self.current_episode_data.append(step_data)
        
        self.current_obs = obs2
        self.episode_reward += reward
        self.episode_steps += 1
        
        if done:
            episode_info = {
                'episode_data': self.current_episode_data.copy(),
                'total_reward': self.episode_reward,
                'total_steps': self.episode_steps,
                'game': self.game_name,
            }
            self.reset()
            return True, episode_info
        
        return False, {}
    
    def sample_action(self) -> int:
        return self.env.action_space.sample() if self.env else 0
    
    def close(self):
        if self.env is not None:
            self.env.close()


class AtariDataGenerator:
    """Atari数据生成器"""
    
    def __init__(self, 
                 num_envs: int,
                 num_episodes_per_env: int,
                 game_list: List[str],
                 save_dir: str,
                 process_id: int,
                 max_memory_mb: int = 500):
        
        self.num_envs = num_envs
        self.num_episodes_per_env = num_episodes_per_env
        self.process_id = process_id
        self.max_memory_mb = max_memory_mb
        self.game_list = game_list if game_list else ["Seaquest-v4"]
        
        self.interrupt_handler = InterruptHandler()
        self.data_saver = SafeDataSaver(save_dir, process_id)
        
        self._allocate_games_to_environments()
        self._create_environments()
        
        print(f"进程 {self.process_id}: 成功创建 {len(self.environments)} 个环境")
        
    def _allocate_games_to_environments(self):
        """将游戏随机分配到环境"""
        self.env_games = np.random.choice(self.game_list, size=self.num_envs).tolist()
    
    def _create_environments(self):
        self.environments = [AtariEnvironment(game, i, self.process_id) for i, game in enumerate(self.env_games)]

    def _save_episode_data(self, episode_info: Dict, episode_id: int) -> Optional[str]:
        """
        处理单个回合的数据，将其扁平化并保存为独立的PKL文件。
        """
        try:
            # 扁平化数据
            all_observations = [step['obs'] for step in episode_info['episode_data']]
            all_actions = [step['action'] for step in episode_info['episode_data']]
            all_rewards = [step['reward'] for step in episode_info['episode_data']]
            all_terminals = [step['done'] for step in episode_info['episode_data']]

            flat_data = {
                'observations': np.array(all_observations, dtype=np.uint8),
                'actions': np.array(all_actions, dtype=np.int64),
                'rewards': np.array(all_rewards, dtype=np.float32), 
                'done': np.array(all_terminals, dtype=bool),
            }

            # 创建元数据和文件名
            metadata = {
                'game': episode_info['game'],
                'total_steps': episode_info['total_steps'],
                'total_reward': episode_info['total_reward'],
                'process_id': self.process_id,
                'episode_id_in_process': episode_id,
            }
            flat_data['metadata'] = metadata

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            game_name_sanitized = episode_info['game'].replace('-', '_') # Sanitize for filename
            filename = f"{game_name_sanitized}_p{self.process_id}_e{episode_id}_{timestamp}.pkl"
            filepath = self.data_saver.save_dir / filename

            # 保存文件
            with open(filepath, 'wb') as f:
                pickle.dump(flat_data, f)
            
            return str(filepath)
        except Exception as e:
            print(f"进程 {self.process_id} 保存回合 {episode_id} 数据失败: {e}")
            return None

    def generate_data(self) -> List[str]:
        """
        生成数据，并将每个回合保存为独立文件。
        """
        print(f"进程 {self.process_id} 开始生成数据: {self.num_envs} 环境 × {self.num_episodes_per_env} 回合")
        
        replay_data, episodes_completed_total = self.data_saver.load_checkpoint()
        saved_filepaths = []

        episodes_per_env = [0] * self.num_envs
        total_target = self.num_envs * self.num_episodes_per_env
        
        if episodes_completed_total > 0:
            for ep_info in replay_data:
                env_id = ep_info.get('env_id', 0)
                if env_id < self.num_envs:
                    episodes_per_env[env_id] +=1
            print(f"进程 {self.process_id} 从检查点恢复，已完成 {episodes_completed_total} 回合")
        
        try:
            while episodes_completed_total < total_target and not self.interrupt_handler.interrupted:
                active_envs_indices = [i for i, count in enumerate(episodes_per_env) if count < self.num_episodes_per_env]
                if not active_envs_indices:
                    break

                for env_idx in active_envs_indices:
                    env = self.environments[env_idx]
                    action = env.sample_action()
                    episode_done, episode_info = env.step(action)
                    
                    if episode_done:
                        episode_info['env_id'] = env_idx
                        replay_data.append(episode_info)
                        episodes_per_env[env_idx] += 1
                        episodes_completed_total += 1
                        
                        print(f"进程 {self.process_id} | 环境 {env_idx+1} | {episode_info['game']} | 完成 {episodes_per_env[env_idx]}/{self.num_episodes_per_env} 回合 | 奖励: {episode_info['total_reward']:.2f} | 总进度: {episodes_completed_total}/{total_target}")

                        # --- 新增：立即保存每个回合的数据 ---
                        filepath = self._save_episode_data(episode_info, episodes_completed_total)
                        if filepath:
                            saved_filepaths.append(filepath)
                        # --- 修改结束 ---

                        if episodes_completed_total > 0 and episodes_completed_total % self.data_saver.checkpoint_interval == 0:
                            self.data_saver.save_checkpoint(replay_data, episodes_completed_total)
        
        finally:
            self.close()
        
        print(f"进程 {self.process_id}: 数据采集完成。总共保存了 {len(saved_filepaths)} 个独立的回合文件。")
        
        # 清理临时检查点文件
        if self.data_saver.temp_file.exists():
            self.data_saver.temp_file.unlink()

        return saved_filepaths
    
    def close(self):
        for env in self.environments:
            env.close()

def generate_data_worker(args):
    """多进程工作函数，捕获键盘中断以允许主进程处理"""
    try:
        process_id, num_envs, num_episodes, games, save_dir, seed = args
        if seed is not None:
            np.random.seed(seed + process_id)
        
        generator = AtariDataGenerator(
            num_envs=num_envs,
            num_episodes_per_env=num_episodes,
            game_list=games,
            save_dir=save_dir,
            process_id=process_id
        )
        return generator.generate_data()
    except KeyboardInterrupt:
        print(f"进程 {process_id} 收到中断，将退出。")
        return [] # 返回空列表
    except Exception as e:
        print(f"进程 {process_id} 发生未处理异常: {e}")
        traceback.print_exc()
        return [] # 返回空列表


def main():
    parser = argparse.ArgumentParser(description='Atari数据生成工具')
    parser.add_argument('--mode', type=str, default='generate', help='此脚本仅支持generate模式')
    parser.add_argument('--num_processes', type=int, default=32, help='使用的进程数')
    parser.add_argument('--total_episodes', type=int, default=32000, help='所有进程总共要生成的回合数')
    parser.add_argument('--games', nargs='+', default=["Seaquest-v4", "Riverraid-v4","ChopperCommand-v4"], help='游戏列表')
    parser.add_argument('--save_dir', type=str, default='../Data/replay_data', help='数据保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    if args.mode != 'generate':
        print("此脚本仅支持 'generate' 模式。")
        return

    episodes_per_process = (args.total_episodes + args.num_processes - 1) // args.num_processes
    envs_per_process = 1
    num_episodes_per_env = episodes_per_process

    process_args = [
        (i, envs_per_process, num_episodes_per_env, args.games, args.save_dir, args.seed)
        for i in range(args.num_processes)
    ]

    print(f"启动 {args.num_processes} 个进程，总共生成约 {args.total_episodes} 个回合...")

    start_time = time.time()
    pool = Pool(processes=args.num_processes)

    try:
        results = pool.map(generate_data_worker, process_args)
        pool.close()
        pool.join()
        
        print(f"\n所有进程正常完成。总耗时: {time.time() - start_time:.2f} 秒")
        
        # 将返回的列表的列表 (list of lists) 扁平化
        all_generated_files = [filepath for sublist in results for filepath in sublist]

        print(f"总共生成了 {len(all_generated_files)} 个文件:")
        for filepath in all_generated_files:
            print(f"- {filepath}")

    except KeyboardInterrupt:
        print("\n主进程收到键盘中断 (Ctrl+C)，正在终止所有子进程...")
        pool.terminate()
        pool.join()
        print("所有进程已终止。程序退出。")
        sys.exit(1)

if __name__ == '__main__':
    if sys.platform.startswith('win') or sys.platform == 'darwin':
        mp.set_start_method('spawn', force=True)
    main()
