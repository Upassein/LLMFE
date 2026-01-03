"""
测试specification文件是否能正常执行
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 模拟LLM-FE的执行环境
sys.path.insert(0, '/data/cqj-project/wind-power-forecast-mlflow-1')
sys.path.insert(0, '/data/cqj-project/LLM-FE/LLMFE')

# 加载数据
print("加载数据...")
df = pd.read_csv('data/windpower_SD_W_B_ecmwf_ifs025.csv', nrows=1000)
print(f"数据形状: {df.shape}")

# 分离X和y
X = df.drop(columns=['power'])
y = df['power'].values

# 构造data字典（模拟LLM-FE的输入）
data = {
    'inputs': X,
    'outputs': y
}

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X columns: {list(X.columns)[:5]}...")

# 定义一个简单的modify_features（baseline）
def modify_features(df_input):
    """Baseline: 不做任何修改"""
    return df_input.copy()

# 执行evaluate函数
print("\n开始测试evaluate函数...")

try:
    # 导入需要的模块
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from core.models.base_lgbm import BaseLGBM
    from utils.grid_template import (
        extract_grid_template,
        apply_template_to_grid,
        merge_all_grids
    )

    print("✓ 所有导入成功")

    # Step 1: Extract template
    inputs = data['inputs']
    outputs = data['outputs']

    print(f"\nStep 1: Extract grid1 template...")
    template_df = extract_grid_template(inputs, grid_id=1, source=None)
    print(f"  Template shape: {template_df.shape}")
    print(f"  Template columns: {list(template_df.columns)}")

    # Step 2: Apply modify_features
    print(f"\nStep 2: Apply modify_features...")
    engineered_template = modify_features(template_df)
    print(f"  Engineered template shape: {engineered_template.shape}")

    # Step 3: Apply to all grids
    print(f"\nStep 3: Apply to all 4 grids...")
    all_grids_features = []
    for grid_id in [1, 2, 3, 4]:
        grid_features = apply_template_to_grid(
            template_df=engineered_template,
            original_inputs=inputs,
            grid_id=grid_id,
            modify_features_func=modify_features,
            source=None
        )
        all_grids_features.append(grid_features)
        print(f"  Grid {grid_id}: {grid_features.shape}")

    # Step 4: Extract time features
    print(f"\nStep 4: Extract time features...")
    time_features = inputs[['hour_sin', 'hour_cos',
                            'day_of_week_sin', 'day_of_week_cos',
                            'month_sin', 'month_cos']]
    print(f"  Time features shape: {time_features.shape}")

    # Step 5: Merge
    print(f"\nStep 5: Merge all grids + time features...")
    X = merge_all_grids(all_grids_features, time_features)
    print(f"  Final X shape: {X.shape}")

    # Convert y
    y = pd.Series(outputs, index=X.index, name='power')
    print(f"  y shape: {y.shape}")

    # Preprocessing
    print(f"\nPreprocessing...")
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'string':
            X[col] = pd.Categorical(X[col]).codes

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    print(f"  After preprocessing: {X.shape}")

    # Test with single fold
    print(f"\nTesting with 1 fold (instead of 4)...")
    kf = KFold(n_splits=2, shuffle=False)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        print(f"\n  Fold {fold+1}:")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print(f"    Train: {X_train.shape}, Test: {X_test.shape}")

        # Create model
        model = BaseLGBM(
            model_name='llmfe_test',
            weather_source='combined',
            params={
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.03,
                'num_leaves': 31,
                'max_depth': 8,
                'min_child_samples': 50,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'verbose': -1
            }
        )

        # Train
        print(f"    Training...")
        model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            num_boost_round=100,  # Reduced for testing
            early_stopping_rounds=20
        )

        # Predict
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"    RMSE: {rmse:.4f} MW")

        # Only test 1 fold
        break

    print("\n" + "="*60)
    print("SUCCESS - Specification test passed!")
    print("="*60)

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
