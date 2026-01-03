#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
服务器端调试脚本 - 检查specification_windpower是否能正常执行
在服务器上运行: python debug_server.py
"""

import sys
import traceback
import pandas as pd
import numpy as np

print("="*80)
print("开始调试 - 检查LLM-FE windpower specification")
print("="*80)

# ============================================================================
# 步骤1: 检查路径和导入
# ============================================================================
print("\n[步骤1] 检查sys.path和导入...")

# 服务器路径 - 用户需要根据实际情况调整
WIND_POWER_PATH = '/data/cqj-project/wind-power-forecast-mlflow-1'
LLMFE_PATH = '/data/cqj-project/LLM-FE/LLMFE'

# 尝试添加路径
for path in [WIND_POWER_PATH, LLMFE_PATH]:
    if path not in sys.path:
        sys.path.insert(0, path)
        print(f"  添加路径: {path}")

# 尝试导入BaseLGBM
try:
    from models.base_lgbm import BaseLGBM
    print("  [成功] 导入 BaseLGBM")
except Exception as e:
    print(f"  [失败] 导入 BaseLGBM: {e}")
    traceback.print_exc()
    sys.exit(1)

# 尝试导入grid_template
try:
    from utils.grid_template import extract_grid_template, apply_template_to_grid, merge_all_grids
    print("  [成功] 导入 grid_template 函数")
except Exception as e:
    print(f"  [失败] 导入 grid_template: {e}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 步骤2: 加载数据
# ============================================================================
print("\n[步骤2] 加载数据...")

DATA_PATH = '/data/cqj-project/wind-power-forecast-mlflow-1/data/sdwpf_bw_ss_cleaned_all_sources_with_target.csv'

try:
    df = pd.read_csv(DATA_PATH)
    print(f"  [成功] 加载数据: {df.shape}")
    print(f"  列数: {len(df.columns)}")
    print(f"  前5列: {df.columns[:5].tolist()}")
except Exception as e:
    print(f"  [失败] 加载数据: {e}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 步骤3: 准备数据
# ============================================================================
print("\n[步骤3] 准备训练数据...")

try:
    # 检查target列
    if 'target' not in df.columns:
        print("  [失败] 找不到target列")
        sys.exit(1)

    # 分离X和y
    X = df.drop(columns=['target'])
    y = df['target']

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  y范围: [{y.min():.2f}, {y.max():.2f}]")

    # 使用前80%作为训练,后20%作为验证(简单分割)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"  训练集: {X_train.shape}, 验证集: {X_val.shape}")

except Exception as e:
    print(f"  [失败] 准备数据: {e}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 步骤4: 测试grid_template提取
# ============================================================================
print("\n[步骤4] 测试grid_template提取...")

try:
    # 提取grid1模板
    template = extract_grid_template(X_train, grid_id=1, source=None)
    print(f"  [成功] 提取模板: {template.shape}")
    print(f"  模板列: {template.columns.tolist()}")

    # 检查是否包含时间特征
    time_features = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']
    has_time = all(tf in template.columns for tf in time_features)
    print(f"  包含时间特征: {has_time}")

except Exception as e:
    print(f"  [失败] 提取grid模板: {e}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 步骤5: 测试modify_features函数(baseline)
# ============================================================================
print("\n[步骤5] 测试baseline modify_features...")

def baseline_modify_features(df_input):
    """最简单的baseline - 直接返回输入"""
    return df_input.copy()

try:
    # 在模板上应用baseline
    engineered_template = baseline_modify_features(template)
    print(f"  [成功] Baseline特征工程: {engineered_template.shape}")

    # 应用到所有4个grid
    all_grids = []
    for grid_id in [1, 2, 3, 4]:
        grid_features = apply_template_to_grid(
            template_df=engineered_template,
            original_inputs=X_train,
            grid_id=grid_id,
            modify_features_func=baseline_modify_features,
            source=None
        )
        all_grids.append(grid_features)
        print(f"  Grid {grid_id}: {grid_features.shape}")

    # 合并
    time_cols = X_train[time_features]
    X_engineered = merge_all_grids(all_grids, time_cols)
    print(f"  [成功] 合并所有grid: {X_engineered.shape}")

except Exception as e:
    print(f"  [失败] 应用grid模板: {e}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 步骤6: 测试BaseLGBM训练
# ============================================================================
print("\n[步骤6] 测试BaseLGBM训练...")

try:
    model = BaseLGBM()

    # 训练
    print("  开始训练...")
    model.fit(X_engineered, y_train)
    print("  [成功] 训练完成")

    # 应用到验证集
    val_template = extract_grid_template(X_val, grid_id=1, source=None)
    val_engineered_template = baseline_modify_features(val_template)

    val_grids = []
    for grid_id in [1, 2, 3, 4]:
        grid_features = apply_template_to_grid(
            template_df=val_engineered_template,
            original_inputs=X_val,
            grid_id=grid_id,
            modify_features_func=baseline_modify_features,
            source=None
        )
        val_grids.append(grid_features)

    val_time_cols = X_val[time_features]
    X_val_engineered = merge_all_grids(val_grids, val_time_cols)

    # 预测
    y_pred = model.predict(X_val_engineered)
    print(f"  [成功] 预测完成: {y_pred.shape}")

    # 计算指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)

    print(f"\n  验证集性能:")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE:  {mae:.4f}")
    print(f"    R2:   {r2:.4f}")

except Exception as e:
    print(f"  [失败] BaseLGBM训练: {e}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("调试完成 - 所有检查通过!")
print("="*80)
print("\n如果这个脚本能成功运行,说明:")
print("  1. 导入路径正确")
print("  2. 数据加载正常")
print("  3. grid_template函数工作正常")
print("  4. BaseLGBM可以训练")
print("\n如果LLM-FE仍然失败,可能是:")
print("  - specification_windpower.txt中的evaluate函数有问题")
print("  - LLM-FE的数据传递格式不匹配")
print("  - 其他LLM-FE框架相关的问题")
print("\n建议: 检查LLM-FE的日志,看initial evaluation的详细错误")
