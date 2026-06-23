import pickle
import numpy as np
import os
import gzip

# 你的文件路径
file_path = "/data0/source/record_fxs/PFLlib-master/dataset/DGA/dga_dataset.gz"

print(f"📂 正在检查文件: {file_path}")

if not os.path.exists(file_path):
    print("❌ 错误: 文件不存在！")
    exit()

data = None
load_method = ""

# 尝试方法 1: 作为 GZIP 读取 (最有可能)
try:
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    load_method = "GZIP"
except (OSError, gzip.BadGzipFile, pickle.UnpicklingError):
    # 尝试方法 2: 作为普通二进制文件读取
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        load_method = "Normal Binary"
    except Exception as e:
        print(f"❌ 读取失败: 两种方法都无法读取。最后一次报错: {e}")
        exit()

print(f"✅ 读取成功！文件格式: {load_method}")
print("=" * 50)

# 解析数据结构
try:
    print(f"数据顶层类型: {type(data)}")

    # 1. 如果是字典 (Dict) - 最常见情况
    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"包含的 Keys: {keys}")
        
        for k in keys:
            v = data[k]
            print(f"\n🔹 Key: '{k}'")
            print(f"   类型: {type(v)}")
            
            # 如果是 Numpy 数组
            if hasattr(v, 'shape'):
                print(f"   Shape: {v.shape}")
            # 如果是列表/元组
            elif isinstance(v, (list, tuple)):
                print(f"   长度: {len(v)}")
                if len(v) > 0:
                    # 检查里面是不是还有数据
                    print(f"   首元素类型: {type(v[0])}")
                    if hasattr(v[0], 'shape'):
                        print(f"   首元素 Shape: {v[0].shape}")

    # 2. 如果是元组或列表 (Tuple/List)
    elif isinstance(data, (list, tuple)):
        print(f"列表/元组长度: {len(data)}")
        if len(data) >= 2:
            print("推测结构可能为 (X, y)")
            if hasattr(data[0], 'shape'):
                print(f"Element 0 Shape: {data[0].shape}")
            if hasattr(data[1], 'shape'):
                print(f"Element 1 Shape: {data[1].shape}")

    # 3. 这里的 Numpy 数组
    elif hasattr(data, 'shape'):
        print(f"这是一个 Numpy 数组, Shape: {data.shape}")

except Exception as e:
    print(f"解析数据结构时出错: {e}")