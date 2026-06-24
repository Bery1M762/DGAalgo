import os
import gzip
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def process_to_rgb(input_file, output_name='dga_rgb_dataset.gz', eps=1e-6, channels_first=True):
    """
    将单个样本的 5 种气体浓度向量 g = [H2, CH4, C2H6, C2H4, C2H2]
    映射为尺寸为 5x5x3 的 RGB 特征张量（或 (3,5,5) 的 channels-first 张量），并将全数据集按 7:3 划分后用 gzip+pickle 保存。

    通道定义：
      - R_{i,j} = g_i / (g_i + g_j + eps)
      - G_{i,j} = ln(1 + |g_i - g_j|) ，随后全矩阵除以 max(G) 归一化至 [0,1]
      - B_{i,j} = (g_i * g_j) / max(g)^2

    参数:
      - input_file: 支持 .xlsx 或 .csv，必须包含列 ['H2','CH4','C2H6','C2H4','C2H2'] 和 '故障类型'
      - output_name: 输出 gzip 文件名
      - eps: 防止除零的小常数
      - channels_first: 如果 True，返回形状 (N,3,5,5)，否则 (N,5,5,3)
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到文件: {input_file}")

    print("读取数据...")
    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)

    gases = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']
    for g in gases:
        if g not in df.columns:
            raise KeyError(f"输入文件缺少所需列: {g}")

    # 标签映射
    unique_labels = df['故障类型'].unique()
    label_map = {label: i for i, label in enumerate(unique_labels)}
    Y = df['故障类型'].map(label_map).values

    X_list = []
    print("生成 RGB 特征张量...")
    for idx, row in df[gases].iterrows():
        g = row.values.astype(np.float64)

        # 防止负值或 NaN（视数据而定，此处仅做保底）
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

        # R 通道: g_i / (g_i + g_j + eps)
        denom = g[:, None] + g[None, :] + eps
        R = g[:, None] / denom

        # G 通道: ln(1 + |g_i - g_j|) 并全局归一化
        G = np.log1p(np.abs(g[:, None] - g[None, :]))
        maxG = G.max()
        if maxG > 0:
            G = G / maxG

        # B 通道: (g_i * g_j) / max(g)^2
        maxg = g.max()
        if maxg > 0:
            B = (g[:, None] * g[None, :]) / (maxg ** 2)
        else:
            B = np.zeros((5, 5), dtype=np.float64)

        # 合并为 5x5x3 (R,G,B)
        rgb = np.stack([R, G, B], axis=-1).astype(np.float32)

        if channels_first:
            rgb = np.transpose(rgb, (2, 0, 1))  # -> (3,5,5)

        X_list.append(rgb)

    X = np.array(X_list, dtype=np.float32)

    # 划分数据集
    print("划分训练/测试集 (8:2)...")
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    dataset = {
        'train': (x_train, y_train),
        'test': (x_test, y_test),
        'label_map': label_map,
        'channels_first': channels_first,
        'gases': gases
    }

    with gzip.open(output_name, 'wb') as f:
        pickle.dump(dataset, f)

    print("已保存:", os.path.abspath(output_name))
    print(f"训练样本: {len(x_train)}, 测试样本: {len(x_test)}")
    return dataset


if __name__ == '__main__':
    import sys

    default_path = r'./data.xlsx'
    input_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    try:
        process_to_rgb(input_path)
    except Exception as e:
        print('处理失败:', e)
