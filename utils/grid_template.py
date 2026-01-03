"""
Grid Template Utilities

辅助函数，用于将多grid的数据转换为单grid模板，让LLM进行特征工程，
然后将工程结果应用回所有grid。
"""
import pandas as pd
import numpy as np
from typing import List, Tuple


def extract_grid_template(df_full: pd.DataFrame, grid_id: int = 1, source: str = None) -> pd.DataFrame:
    """
    从完整的多grid数据中提取单个grid的模板

    自动识别气象源和特征结构，支持：
    - ecmwf_ifs025: 5个特征 (10m)
    - gfs_global: 9个特征 (10m + 80m + 120m)
    - icon_global: 11个特征 (10m + 80m + 120m + 180m)

    Args:
        df_full: 完整DataFrame
        grid_id: 要提取的grid编号 (1, 2, 3, 或 4)
        source: 气象源名称（如 'ecmwf_ifs025'），如果为None则自动检测

    Returns:
        单grid的模板DataFrame
    """
    # 定义时间特征（不属于任何grid，是全局的）
    time_features = [
        'hour_sin', 'hour_cos',
        'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos'
    ]

    # 自动检测气象源
    if source is None:
        for col in df_full.columns:
            if '_grid' in col:
                source = col.split('_grid')[0]
                break
        if source is None:
            # 如果没有找到 _grid，说明已经是 template 格式，直接返回
            # (这发生在 LLM-FE 第二次调用时，inputs 已经是 template)
            return df_full

    # 查找该grid的所有气象列
    grid_prefix = f"{source}_grid{grid_id}_"
    grid_cols = {}

    for col in df_full.columns:
        if col.startswith(grid_prefix) and col not in time_features:
            # 去掉前缀，得到简化的列名
            simple_name = col[len(grid_prefix):]
            grid_cols[col] = simple_name

    if not grid_cols:
        raise ValueError(f"未找到grid{grid_id}的列（前缀: {grid_prefix}）")

    # 提取数据
    template_df = pd.DataFrame(index=df_full.index)

    # 添加气象变量（重命名，去掉grid前缀）
    for full_col, simple_col in grid_cols.items():
        template_df[simple_col] = df_full[full_col].copy()

    # 添加时间特征
    for time_col in time_features:
        if time_col in df_full.columns:
            template_df[time_col] = df_full[time_col].copy()
        else:
            raise ValueError(f"Time feature {time_col} not found in input DataFrame")

    return template_df


def apply_template_to_grid(
    template_df: pd.DataFrame,
    original_inputs: pd.DataFrame,
    grid_id: int,
    modify_features_func,
    source: str = None
) -> pd.DataFrame:
    """
    将工程化的模板应用到指定grid

    方案：重新执行法
    1. 从original_inputs提取该grid的原始5列数据
    2. 添加时间特征（6列）-> 得到11列
    3. 重新调用 modify_features_func()
    4. 重命名列，加上grid前缀
    5. 返回工程化的grid特征（带前缀）

    Args:
        template_df: modify_features()的输出（11+N列）
            包含原始11列 + LLM新增的N列
        original_inputs: 完整的26列原始数据
        grid_id: 要应用到的grid编号 (1, 2, 3, 或 4)
        modify_features_func: modify_features函数对象
        source: 气象源名称

    Returns:
        该grid的工程化特征（5+N列，带grid前缀）
        注意：不包含时间特征（时间特征是全局的，会在最后统一添加）
    """
    # 0. 如果source为None，自动检测
    if source is None:
        for col in original_inputs.columns:
            if '_grid' in col:
                source = col.split('_grid')[0]
                break
        if source is None:
            # 调试信息
            print(f"[ERROR] 无法自动检测气象源")
            print(f"  original_inputs.shape: {original_inputs.shape}")
            print(f"  original_inputs.columns: {original_inputs.columns.tolist()}")
            raise ValueError("无法自动检测气象源，请手动指定source参数")

    # 1. 提取该grid的原始数据（11列）
    grid_template = extract_grid_template(original_inputs, grid_id=grid_id, source=source)

    # 2. 重新调用modify_features生成工程特征
    engineered = modify_features_func(grid_template)

    # 3. 识别新增的特征（排除时间特征）
    time_features = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']

    # 只保留气象相关的列（原始5列 + 新增特征），去掉时间特征
    engineered_weather_only = engineered.drop(columns=time_features, errors='ignore')

    # 4. 重命名所有列，加上grid前缀
    grid_prefix = f"{source}_grid{grid_id}_"
    renamed_cols = {}
    for col in engineered_weather_only.columns:
        # 如果列名已经有grid前缀（理论上不应该），先去掉
        if col.startswith(source):
            # 去掉可能的旧前缀
            simple_name = col.split('_', 2)[-1] if '_grid' in col else col
        else:
            simple_name = col

        renamed_cols[col] = f"{grid_prefix}{simple_name}"

    result = engineered_weather_only.rename(columns=renamed_cols)

    return result


def get_feature_names_from_template(template_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    从工程化模板中识别原始特征和新增特征

    Args:
        template_df: modify_features()的输出

    Returns:
        (original_features, new_features)
    """
    # 定义原始特征
    original_weather = [
        'relative_humidity_2m',
        'wind_speed_10m',
        'wind_direction_10m',
        'pressure_msl',
        'surface_pressure'
    ]

    time_features = [
        'hour_sin', 'hour_cos',
        'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos'
    ]

    original_all = original_weather + time_features

    # 新增特征 = 所有列 - 原始列
    new_features = [col for col in template_df.columns if col not in original_all]

    return original_all, new_features


def merge_all_grids(
    grid_features_list: List[pd.DataFrame],
    time_features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    合并所有grid的特征 + 时间特征

    Args:
        grid_features_list: 4个grid的特征列表，每个是DataFrame
        time_features_df: 时间特征DataFrame (6列)

    Returns:
        合并后的完整特征DataFrame
    """
    # 横向拼接所有grid
    all_grids = pd.concat(grid_features_list, axis=1)

    # 添加时间特征（只加一次）
    result = pd.concat([all_grids, time_features_df], axis=1)

    return result


if __name__ == "__main__":
    # 简单测试
    print("Grid template utilities loaded successfully!")

    # 创建测试数据
    test_data = pd.DataFrame({
        'ecmwf_ifs025_grid1_relative_humidity_2m': [80.0, 75.0],
        'ecmwf_ifs025_grid1_wind_speed_10m': [5.0, 6.0],
        'ecmwf_ifs025_grid1_wind_direction_10m': [180.0, 200.0],
        'ecmwf_ifs025_grid1_pressure_msl': [1013.0, 1012.0],
        'ecmwf_ifs025_grid1_surface_pressure': [1010.0, 1009.0],
        'ecmwf_ifs025_grid2_relative_humidity_2m': [78.0, 76.0],
        'ecmwf_ifs025_grid2_wind_speed_10m': [5.2, 6.1],
        'ecmwf_ifs025_grid2_wind_direction_10m': [185.0, 205.0],
        'ecmwf_ifs025_grid2_pressure_msl': [1013.5, 1012.5],
        'ecmwf_ifs025_grid2_surface_pressure': [1010.5, 1009.5],
        'hour_sin': [0.5, 0.7],
        'hour_cos': [0.8, 0.6],
        'day_of_week_sin': [0.3, 0.4],
        'day_of_week_cos': [0.9, 0.85],
        'month_sin': [0.1, 0.15],
        'month_cos': [0.99, 0.98]
    })

    # 测试extract_grid_template
    template = extract_grid_template(test_data, grid_id=1)
    print(f"\n提取的模板列: {list(template.columns)}")
    print(f"模板形状: {template.shape}")

    print("\n✅ 所有测试通过!")
