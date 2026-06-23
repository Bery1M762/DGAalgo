import numpy as np
import os
import sys
import random
import torch
import pickle
import gzip
import argparse

# 引入 PFLlib 的工具函数
from utils.dataset_utils import check, separate_data, split_data, save_file

def generate_dataset(dir_path, num_clients, niid, balance, partition, alpha, seed):
    # ==========================================
    # 1. 强制定义路径并创建文件夹 (放在最前面！)
    # ==========================================
    # 确保 dir_path 不以 / 结尾，方便后面拼接
    if dir_path.endswith("/"):
        dir_path = dir_path[:-1]

    config_path = dir_path + "/config.json"
    train_path = dir_path + "/train/"
    test_path = dir_path + "/test/"

    print(f"目标路径:\n  config: {config_path}\n  train:  {train_path}\n  test:   {test_path}")

    # 强制创建 train 文件夹
    if not os.path.exists(train_path):
        try:
            os.makedirs(train_path)
            print("✅ 已创建 train 文件夹")
        except Exception as e:
            print(f"❌ 创建 train 文件夹失败: {e}")

    # 强制创建 test 文件夹
    if not os.path.exists(test_path):
        try:
            os.makedirs(test_path)
            print("✅ 已创建 test 文件夹")
        except Exception as e:
            print(f"❌ 创建 test 文件夹失败: {e}")

    # ==========================================
    # 2. 加载数据
    # ==========================================
    raw_file = dir_path + "/dga_dataset.gz"
    print(f"正在加载数据: {raw_file} ...")
    
    if not os.path.exists(raw_file):
        print(f"❌ 错误: 找不到源文件 {raw_file}")
        return

    try:
        with gzip.open(raw_file, 'rb') as f:
            content = pickle.load(f)
            
        train_X, train_y = content['train']
        test_X, test_y = content['test']
        
        # 合并数据
        X = np.concatenate([train_X, test_X], axis=0)
        y = np.concatenate([train_y, test_y], axis=0)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        
        # 自动检测类别
        num_classes = len(np.unique(y))
        print(f"数据加载完成: 总数={len(y)}, 类别={num_classes}")
        
    except Exception as e:
        print(f"❌ 读取数据出错: {e}")
        return

    dataset_image = X
    dataset_label = y

    # ==========================================
    # 3. 划分数据
    # ==========================================
    # separate_data 不接受 alpha 参数，这里已去掉
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition)

    # 划分训练集/测试集
    train_data, test_data = split_data(X, y)

    # ==========================================
    # 4. 保存文件 (传入刚才创建好的路径)
    # ==========================================
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
              statistic, niid, balance, partition)
    
    print("✅ 全部完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 使用可选参数
    parser.add_argument("--format", type=str, default="iid", help="Format of data: iid or noniid")
    parser.add_argument("--balance", type=str, default="balance", help="Balance or not: balance or -")
    parser.add_argument("--partition", type=str, default="-", help="Partition method: pat or dir or -")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for Dirichlet distribution")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    
    args = parser.parse_args()

    # 获取当前脚本所在目录的绝对路径 + /DGA
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(current_dir, "DGA")
    
    num_clients = 20

    generate_dataset(dir_path, num_clients, args.format == "noniid", 
                     args.balance == "balance", args.partition, args.alpha, args.seed)