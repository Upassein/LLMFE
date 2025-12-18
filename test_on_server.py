"""
在服务器上直接测试 evaluate 函数
"""
import sys
sys.path.insert(0, '/data/cqj-project/wind-power-forecast-mlflow-1')

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from core.models.base_lgbm import BaseLGBM

print("="*60)
print("服务器端测试 - 直接调用 evaluate 逻辑")
print("="*60)

# 加载数据
print("\n1. 加载数据...")
df = pd.read_csv('./data/windpower_SD_W_B_ecmwf_ifs025.csv')
print(f"数据形状: {df.shape}")

# 模拟 main.py 的处理
target_attr = 'power'
X = df.convert_dtypes()
y = df[target_attr].to_numpy()
X = X.drop(target_attr, axis=1)
X = X.fillna(0)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}, type: {type(y)}")
print(f"y stats: mean={y.mean():.2f}, std={y.std():.2f}")

# 只用前 5000 条快速测试
X = X.head(5000)
y = y[:5000]
print(f"\n测试用数据: {X.shape}")

# === 开始 evaluate 逻辑 ===
print("\n2. 执行 evaluate 逻辑...")

# Baseline 特征工程（不做任何变换）
def modify_features(df_input):
    return df_input.copy()

X_transformed = modify_features(X)

# 转换 y 为 Series（关键！）
print(f"\n3. 转换 y 为 Series...")
print(f"  转换前: type={type(y)}")
if isinstance(y, np.ndarray):
    y_series = pd.Series(y, index=X_transformed.index, name='power')
else:
    y_series = y
print(f"  转换后: type={type(y_series)}")
print(f"  Series stats: mean={y_series.mean():.2f}, std={y_series.std():.2f}")

# 处理缺失值
X_transformed = X_transformed.replace([np.inf, -np.inf], np.nan)
X_transformed = X_transformed.fillna(0)

# K-Fold CV
print("\n4. 开始 K-Fold CV...")
kf = KFold(n_splits=4, shuffle=False)
scores = []

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_transformed, y_series), 1):
    print(f"\n  === Fold {fold_idx}/4 ===")

    X_train = X_transformed.iloc[train_idx]
    X_test = X_transformed.iloc[test_idx]
    y_train = y_series.iloc[train_idx]
    y_test = y_series.iloc[test_idx]

    print(f"  训练集: X={X_train.shape}, y={y_train.shape}, y type={type(y_train)}")
    print(f"  测试集: X={X_test.shape}, y={y_test.shape}, y type={type(y_test)}")
    print(f"  y_train stats: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
    print(f"  y_test stats: mean={y_test.mean():.2f}, std={y_test.std():.2f}")

    # 创建 BaseLGBM（和 specification 一模一样的参数）
    model = BaseLGBM(
        model_name='test_eval',
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

    try:
        # 训练
        print(f"  开始训练...")
        model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            num_boost_round=100,  # 测试用少点
            early_stopping_rounds=10
        )

        # 预测
        print(f"  开始预测...")
        y_pred = model.predict(X_test)

        print(f"  预测值 type: {type(y_pred)}")
        print(f"  预测值 shape: {y_pred.shape}")
        print(f"  预测值统计: mean={y_pred.mean():.2f}, min={y_pred.min():.2f}, max={y_pred.max():.2f}")

        # 计算 RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"  ✓ RMSE: {rmse:.2f} MW")

        scores.append(-rmse)

    except Exception as e:
        print(f"  ❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        scores.append(-999999)

# 汇总
print("\n" + "="*60)
print("测试完成！")
print("="*60)
avg_score = np.mean(scores)
print(f"平均分数: {avg_score:.2f}")
print(f"平均 RMSE: {-avg_score:.2f} MW")
print(f"各 fold RMSE: {[-s for s in scores]}")

if -avg_score < 50:
    print("\n✅ 测试成功！RMSE 合理（< 50 MW）")
else:
    print(f"\n❌ RMSE 异常大！({-avg_score:.2f} MW)")
    print("这说明有某个环节出了问题，但数据本身是对的。")
