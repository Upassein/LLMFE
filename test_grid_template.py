"""
测试grid template功能

验证特征维度的正确性
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.grid_template import (
    extract_grid_template,
    apply_template_to_grid,
    merge_all_grids
)


def create_test_data(n_rows=100):
    """创建测试数据（模拟26列windpower数据）"""
    np.random.seed(42)

    data = {}

    # 4个grid，每个5个气象变量
    for grid_id in [1, 2, 3, 4]:
        prefix = f"ecmwf_ifs025_grid{grid_id}_"
        data[f"{prefix}relative_humidity_2m"] = np.random.uniform(50, 90, n_rows)
        data[f"{prefix}wind_speed_10m"] = np.random.uniform(0, 20, n_rows)
        data[f"{prefix}wind_direction_10m"] = np.random.uniform(0, 360, n_rows)
        data[f"{prefix}pressure_msl"] = np.random.uniform(1000, 1020, n_rows)
        data[f"{prefix}surface_pressure"] = np.random.uniform(995, 1015, n_rows)

    # 6个时间特征
    data['hour_sin'] = np.sin(2 * np.pi * np.arange(n_rows) / 24)
    data['hour_cos'] = np.cos(2 * np.pi * np.arange(n_rows) / 24)
    data['day_of_week_sin'] = np.sin(2 * np.pi * np.arange(n_rows) / 7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * np.arange(n_rows) / 7)
    data['month_sin'] = np.sin(2 * np.pi * np.arange(n_rows) / 12)
    data['month_cos'] = np.cos(2 * np.pi * np.arange(n_rows) / 12)

    df = pd.DataFrame(data)
    return df


def simple_modify_features(df_input):
    """简单的特征工程函数（用于测试）"""
    df_output = df_input.copy()

    # 添加3个新特征
    df_output['wind_speed_squared'] = df_input['wind_speed_10m'] ** 2
    df_output['wind_speed_cubed'] = df_input['wind_speed_10m'] ** 3

    # 风向分解
    wind_dir_rad = np.deg2rad(df_input['wind_direction_10m'])
    df_output['wind_u'] = df_input['wind_speed_10m'] * np.sin(wind_dir_rad)
    df_output['wind_v'] = df_input['wind_speed_10m'] * np.cos(wind_dir_rad)

    return df_output


def test_extract_grid_template():
    """测试提取grid模板"""
    print("\n" + "="*60)
    print("测试1: extract_grid_template()")
    print("="*60)

    df_full = create_test_data(100)
    print(f"原始数据形状: {df_full.shape}")
    print(f"原始列数: {len(df_full.columns)}")

    # 提取grid1模板
    template = extract_grid_template(df_full, grid_id=1)

    print(f"\n模板数据形状: {template.shape}")
    print(f"模板列数: {len(template.columns)}")
    print(f"模板列名:")
    for i, col in enumerate(template.columns, 1):
        print(f"  {i}. {col}")

    # 验证
    assert template.shape[0] == 100, f"行数错误: {template.shape[0]}"
    assert template.shape[1] == 11, f"列数错误: {template.shape[1]}, 期望11列"

    expected_cols = [
        'relative_humidity_2m',
        'wind_speed_10m',
        'wind_direction_10m',
        'pressure_msl',
        'surface_pressure',
        'hour_sin', 'hour_cos',
        'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos'
    ]

    for col in expected_cols:
        assert col in template.columns, f"缺少列: {col}"

    print("\nOK extract_grid_template 测试通过!")
    return df_full, template


def test_apply_template_to_grid(df_full, template):
    """测试应用模板到grid"""
    print("\n" + "="*60)
    print("测试2: apply_template_to_grid()")
    print("="*60)

    # 应用特征工程到模板
    engineered_template = simple_modify_features(template)
    print(f"工程化模板形状: {engineered_template.shape}")
    print(f"工程化模板列数: {len(engineered_template.columns)}")
    print(f"新增特征数: {len(engineered_template.columns) - 11}")

    # 应用到grid2
    grid2_features = apply_template_to_grid(
        template_df=engineered_template,
        original_inputs=df_full,
        grid_id=2,
        modify_features_func=simple_modify_features,
        source='ecmwf_ifs025'
    )

    print(f"\nGrid2特征形状: {grid2_features.shape}")
    print(f"Grid2特征列数: {len(grid2_features.columns)}")
    print(f"Grid2前5列:")
    for i, col in enumerate(list(grid2_features.columns)[:5], 1):
        print(f"  {i}. {col}")

    # 验证
    assert grid2_features.shape[0] == 100, "行数错误"
    # 应该是 5个原始气象变量 + 4个新增特征 = 9列（不包含时间特征）
    expected_cols_count = 5 + 4  # 原始5列 + 4个新特征
    assert grid2_features.shape[1] == expected_cols_count, \
        f"列数错误: {grid2_features.shape[1]}, 期望{expected_cols_count}"

    # 检查列名是否有grid2前缀
    for col in grid2_features.columns:
        assert col.startswith('ecmwf_ifs025_grid2_'), f"列名缺少grid2前缀: {col}"

    print("\nOK apply_template_to_grid 测试通过!")
    return grid2_features


def test_full_workflow():
    """测试完整工作流"""
    print("\n" + "="*60)
    print("测试3: 完整工作流 (4个grid)")
    print("="*60)

    df_full = create_test_data(100)
    print(f"原始数据: {df_full.shape}")

    # Step 1: 提取grid1模板
    template = extract_grid_template(df_full, grid_id=1)
    print(f"Step 1 - 模板: {template.shape}")

    # Step 2: 应用特征工程
    engineered_template = simple_modify_features(template)
    print(f"Step 2 - 工程化模板: {engineered_template.shape}")

    # Step 3: 应用到所有4个grid
    all_grids_features = []
    for grid_id in [1, 2, 3, 4]:
        grid_features = apply_template_to_grid(
            template_df=engineered_template,
            original_inputs=df_full,
            grid_id=grid_id,
            modify_features_func=simple_modify_features,
            source='ecmwf_ifs025'
        )
        all_grids_features.append(grid_features)
        print(f"  Grid {grid_id}: {grid_features.shape}")

    # Step 4: 提取时间特征
    time_features = df_full[['hour_sin', 'hour_cos',
                             'day_of_week_sin', 'day_of_week_cos',
                             'month_sin', 'month_cos']]
    print(f"Step 4 - 时间特征: {time_features.shape}")

    # Step 5: 合并
    X_final = merge_all_grids(all_grids_features, time_features)
    print(f"Step 5 - 最终特征: {X_final.shape}")

    # 验证
    # 每个grid: 5原始 + 4新增 = 9列
    # 4个grid × 9列 = 36列
    # + 6个时间特征 = 42列
    expected_total = (5 + 4) * 4 + 6
    assert X_final.shape[1] == expected_total, \
        f"最终列数错误: {X_final.shape[1]}, 期望{expected_total}"

    print(f"\n最终特征分布:")
    print(f"  - Grid特征: {(5 + 4) * 4} 列 (每个grid 9列 × 4个grid)")
    print(f"  - 时间特征: 6 列")
    print(f"  - 总计: {X_final.shape[1]} 列")

    # 检查列名
    grid_cols = [col for col in X_final.columns if 'grid' in col]
    time_cols = [col for col in X_final.columns if col.startswith(('hour', 'day', 'month'))]

    print(f"\n列名统计:")
    print(f"  - Grid相关列: {len(grid_cols)}")
    print(f"  - 时间相关列: {len(time_cols)}")

    assert len(grid_cols) == (5 + 4) * 4, "Grid列数不对"
    assert len(time_cols) == 6, "时间列数不对"

    print("\nOK 完整工作流测试通过!")

    return X_final


def test_feature_values():
    """测试特征值的正确性"""
    print("\n" + "="*60)
    print("测试4: 特征值验证")
    print("="*60)

    df_full = create_test_data(10)  # 只用10行，方便验证

    # 提取grid1
    template = extract_grid_template(df_full, grid_id=1)

    # 手动计算grid1的wind_speed_squared
    expected_speed_sq = df_full['ecmwf_ifs025_grid1_wind_speed_10m'] ** 2

    # 通过特征工程生成
    engineered_template = simple_modify_features(template)

    # 验证值是否相等
    actual_speed_sq = engineered_template['wind_speed_squared']

    np.testing.assert_array_almost_equal(
        expected_speed_sq.values,
        actual_speed_sq.values,
        decimal=5,
        err_msg="wind_speed_squared值不匹配"
    )

    print("OK 特征值验证通过!")


if __name__ == "__main__":
    print("="*60)
    print("开始测试 Grid Template 功能")
    print("="*60)

    try:
        # 测试1: 提取模板
        df_full, template = test_extract_grid_template()

        # 测试2: 应用到单个grid
        grid2_features = test_apply_template_to_grid(df_full, template)

        # 测试3: 完整工作流
        X_final = test_full_workflow()

        # 测试4: 特征值验证
        test_feature_values()

        print("\n" + "="*60)
        print("SUCCESS - All tests passed!")
        print("="*60)

    except Exception as e:
        print(f"\nFAIL 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
