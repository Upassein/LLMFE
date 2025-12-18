"""
测试 specification_windpower.txt 的 evaluate 函数
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 添加 Wind 项目路径
sys.path.insert(0, r'E:\VIVADO\wind-power-forecast-mlflow-1')
from core.models.base_lgbm import BaseLGBM

# 加载数据
print("加载数据...")
df = pd.read_csv(r'E:\VIVADO\LLM-FE\LLMFE\data\windpower_SD_W_B_ecmwf_ifs025.csv')
print(f"数据形状: {df.shape}")

# 准备数据（模拟 main.py 的处理）
X = df.iloc[:, :-1]  # 所有列除了最后一列
y = df.iloc[:, -1].to_numpy()  # 最后一列转为 numpy array

print(f"X shape: {X.shape}, type: {type(X)}")
print(f"y shape: {y.shape}, type: {type(y)}")
print(f"y 统计: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}, std={y.std():.2f}")

# 只用前 5000 条测试（快一点）
X = X.head(5000)
y = y[:5000]

# 模拟 evaluate 函数
def modify_features(df_input):
    """简单的特征工程（baseline）"""
    df_output = df_input.copy()
    return df_output

print("\n开始测试 evaluate 逻辑...")

# 应用特征工程
X_transformed = modify_features(X)
print(f"特征工程后: {X_transformed.shape}")

# 转换 y 为 Series（关键修改）
if isinstance(y, np.ndarray):
    y_series = pd.Series(y, index=X_transformed.index, name='power')
    print(f"y 转换为 Series: {type(y_series)}")
else:
    y_series = y

# 处理缺失值和无穷值
X_transformed = X_transformed.replace([np.inf, -np.inf], np.nan)
X_transformed = X_transformed.fillna(0)

# K-Fold CV
print("\n开始 4-Fold CV...")
kf = KFold(n_splits=4, shuffle=False)
scores = []

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_transformed, y_series), 1):
    print(f"\n--- Fold {fold_idx}/4 ---")
    X_train = X_transformed.iloc[train_idx]
    X_test = X_transformed.iloc[test_idx]
    y_train = y_series.iloc[train_idx]
    y_test = y_series.iloc[test_idx]

    print(f"训练集: X={X_train.shape}, y={y_train.shape}, y type={type(y_train)}")
    print(f"测试集: X={X_test.shape}, y={y_test.shape}, y type={type(y_test)}")

    # 创建 BaseLGBM
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
        print("开始训练...")
        model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            num_boost_round=100,  # 测试用少点轮数
            early_stopping_rounds=10
        )

        # 预测
        print("开始预测...")
        y_pred = model.predict(X_test)

        # 计算 RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"RMSE: {rmse:.2f} MW")

        # 检查预测值范围
        print(f"预测值范围: min={y_pred.min():.2f}, max={y_pred.max():.2f}, mean={y_pred.mean():.2f}")
        print(f"真实值范围: min={y_test.min():.2f}, max={y_test.max():.2f}, mean={y_test.mean():.2f}")

        scores.append(-rmse)

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        scores.append(-999999)

# 汇总结果
print("\n" + "="*60)
print("测试完成！")
print("="*60)
avg_score = np.mean(scores)
print(f"平均分数: {avg_score:.2f}")
print(f"平均 RMSE: {-avg_score:.2f} MW")
print(f"各 fold RMSE: {[-s for s in scores]}")

# 判断是否合理
if -avg_score < 50:
    print("✅ RMSE 看起来合理！（< 50 MW）")
else:
    print(f"⚠️  RMSE 偏高！（{-avg_score:.2f} MW，装机容量 99 MW）")
