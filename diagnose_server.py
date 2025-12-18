"""
服务器诊断脚本 - 检查数据和配置是否正确
"""
import pandas as pd
import numpy as np

print("="*60)
print("诊断脚本 - 检查 LLM-FE 配置")
print("="*60)

# 1. 检查数据文件
print("\n1. 检查数据文件...")
try:
    df = pd.read_csv('./data/windpower_SD_W_B_ecmwf_ifs025.csv')
    print(f"✓ 数据加载成功")
    print(f"  - 形状: {df.shape}")
    print(f"  - 列数: {len(df.columns)}")
    print(f"  - 行数: {len(df)}")

    # 检查 power 列
    if 'power' in df.columns:
        power = df['power']
        print(f"\n  Power 列统计:")
        print(f"    - Mean: {power.mean():.2f}")
        print(f"    - Std: {power.std():.2f}")
        print(f"    - Min: {power.min():.2f}")
        print(f"    - Max: {power.max():.2f}")
        print(f"    - 前5个值: {power.head().tolist()}")

        if power.mean() > 100:
            print(f"  ⚠️  警告: Power 均值 {power.mean():.2f} 异常大!")
        elif power.mean() < 1:
            print(f"  ⚠️  警告: Power 均值 {power.mean():.2f} 异常小!")
        else:
            print(f"  ✓ Power 数值范围看起来正常")
    else:
        print(f"  ❌ 未找到 'power' 列!")
        print(f"  列名: {list(df.columns)}")

except FileNotFoundError:
    print("❌ 数据文件不存在: ./data/windpower_SD_W_B_ecmwf_ifs025.csv")
except Exception as e:
    print(f"❌ 加载数据失败: {e}")

# 2. 检查 specification 文件
print("\n2. 检查 specification 文件...")
try:
    with open('./specs/specification_windpower.txt', 'r', encoding='utf-8') as f:
        spec_content = f.read()

    print("✓ Specification 文件存在")

    # 检查关键修改是否存在
    checks = {
        'pd.Series(outputs': 'y 转 Series 的修改',
        'y.iloc[train_idx]': 'Series 索引修改',
        'num_boost_round=500': '500 轮训练',
        'reg_alpha': '正则化参数'
    }

    print("\n  关键代码检查:")
    for pattern, desc in checks.items():
        if pattern in spec_content:
            print(f"    ✓ {desc}")
        else:
            print(f"    ❌ 缺失: {desc}")

except FileNotFoundError:
    print("❌ Specification 文件不存在")
except Exception as e:
    print(f"❌ 读取 specification 失败: {e}")

# 3. 检查 main.py
print("\n3. 检查 main.py 修改...")
try:
    with open('./main.py', 'r', encoding='utf-8') as f:
        main_content = f.read()

    if 'Use ALL data' in main_content:
        print("✓ main.py 已修改（禁用外层 5-Fold）")
    else:
        print("⚠️  main.py 可能未修改（仍使用 5-Fold）")

except Exception as e:
    print(f"❌ 检查 main.py 失败: {e}")

# 4. 模拟 LLM-FE 的数据加载
print("\n4. 模拟 LLM-FE 数据加载流程...")
try:
    df = pd.read_csv('./data/windpower_SD_W_B_ecmwf_ifs025.csv')

    # 模拟 main.py 的处理
    target_attr = df.columns[-1]
    X = df.convert_dtypes()
    y = df[target_attr].to_numpy()
    X = X.drop(target_attr, axis=1)
    X = X.fillna(0)

    print(f"✓ 数据处理完成")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"  - y type: {type(y)}")
    print(f"  - y stats: mean={y.mean():.2f}, std={y.std():.2f}")

    # 检查是否需要转换
    if isinstance(y, np.ndarray):
        y_series = pd.Series(y, index=X.index, name='power')
        print(f"  ✓ y 可以正确转换为 Series")
        print(f"    - Series mean: {y_series.mean():.2f}")

except Exception as e:
    print(f"❌ 模拟数据加载失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("诊断完成!")
print("="*60)
print("\n如果所有检查都通过，但 RMSE 仍然很大，")
print("请检查服务器上 Wind 项目路径是否正确:")
print("  /data/cqj-project/wind-power-forecast-mlflow-1")
