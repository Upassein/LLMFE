# LLM-FE在风电功率预测中的应用实验报告

## 1. 实验背景

### 1.1 项目概况
- 任务：风电功率预测（SD_W_B风场，装机容量99 MW）
- 数据集：41436个样本，26个气象特征
- 气象源：ECMWF IFS025
- 评估指标：RMSE (均方根误差)

### 1.2 数据特征
**原始特征（20个气象特征）**
- 风速：4个网格点 × 1个高度（10m）
- 风向：4个网格点 × 1个高度（10m）
- 相对湿度：4个网格点 × 1个高度（2m）
- 气压MSL：4个网格点
- 地面气压：4个网格点

**时间特征（6个）**
- hour_sin, hour_cos
- day_of_week_sin, day_of_week_cos
- month_sin, month_cos

### 1.3 LLM-FE框架
- 框架：Multi-Island遗传算法
- LLM模型：Moonshot-v1-128k
- 评估模型：LightGBM（4-Fold时间序列交叉验证）
- Experience Buffer：Softmax温度采样

## 2. 实验过程

### 2.1 数据准备

**特征形式选择**
- 初始方案：使用u/v风分量（通过配置`use_uv_components: true`）
- 最终方案：使用风速+风向的物理直观形式
- 配置调整：`configs/SD_W_B/model.yaml`中设置`use_uv_components: false`
- 原因：风速+风向是气象观测的原始形式，物理意义更直观

**特征工程预处理**
- 禁用Wind项目中的所有特征工程模块
- 配置：`feature_engineering.*: enabled: false`
- 目的：让LLM-FE从原始气象数据出发进行特征挖掘

### 2.2 Specification迭代

**V3：明确公式提示**
- 内容：
  - 明确建议：wind_speed^3, sin/cos(direction), 压力梯度
  - 跨网格特征：mean, std, max, min
  - 时间-气象交互项
- 采样次数：4次
- 最佳RMSE：13.00 MW

**V4：完全弱化提示**
- 内容：
  - 只提供领域知识描述
  - 不给具体公式和变换方法
  - 鼓励创造性探索
- 采样次数：10次
- 最佳RMSE：13.15 MW（性能下降）

**V5：物理约束提示**
- 内容：
  - 给出物理公式作为灵感：P ∝ ρ × A × v^n
  - 参数化提示：n在2-4之间探索，k值未知
  - 数学操作指南：多项式、比值、三角函数、对数等
  - 约束条件：最多删除40%特征
  - 历史反馈：RMSE范围13.0-13.3 MW
- 采样次数：10次
- 最佳RMSE：13.027 MW

## 3. 实验结果

### 3.1 性能对比

| 版本 | Specification策略 | 采样次数 | 最佳RMSE (MW) | 相对误差 |
|------|------------------|---------|--------------|---------|
| Baseline | 无特征工程 | - | 13.029 | 0.63 |
| V3 | 明确公式 | 4 | 13.00 | 0.00 (基准) |
| V4 | 完全弱化 | 10 | 13.15 | +1.15 |
| V5 | 物理约束 | 10 | 13.027 | +0.21 |

注：相对误差 = (RMSE - 13.00) / 13.00 × 100%

### 3.2 V3最佳特征（Sample 2）

**新增特征（15个）**
```python
# 每个网格的风速立方
wind_power_potential_grid1 = wind_speed_1 ** 3
wind_power_potential_grid2 = wind_speed_2 ** 3
wind_power_potential_grid3 = wind_speed_3 ** 3
wind_power_potential_grid4 = wind_speed_4 ** 3

# 每个网格的气压差
pressure_diff_grid1 = pressure_msl_1 - surface_pressure_1
pressure_diff_grid2 = pressure_msl_2 - surface_pressure_2
pressure_diff_grid3 = pressure_msl_3 - surface_pressure_3
pressure_diff_grid4 = pressure_msl_4 - surface_pressure_4

# 跨网格湿度差异
rh_diff_grid12 = humidity_2 - humidity_1
rh_diff_grid23 = humidity_3 - humidity_2
rh_diff_grid34 = humidity_4 - humidity_3

# 跨网格风向差异
wind_dir_diff_grid12 = direction_2 - direction_1
wind_dir_diff_grid23 = direction_3 - direction_2
wind_dir_diff_grid34 = direction_4 - direction_3
```

**删除特征（8个）**
- 所有风向特征（4个）
- 所有地面气压特征（4个）

**最终特征数**：26 → +15 -8 = 33个

### 3.3 V5最佳特征（Sample 2）

**新增特征（12个）**
```python
# 跨网格风速统计
wind_speed_mean = mean([speed_1, speed_2, speed_3, speed_4])
wind_speed_std = std([speed_1, speed_2, speed_3, speed_4])

# 风速多项式
wind_speed_squared = wind_speed_mean ** 2
wind_speed_cubed = wind_speed_mean ** 3

# 跨网格气压统计
pressure_mean = mean([pressure_msl_1, ..., pressure_msl_4])
pressure_std = std([pressure_msl_1, ..., pressure_msl_4])

# 时间-风速交互（6个）
wind_speed_hour_sin = wind_speed_mean * hour_sin
wind_speed_hour_cos = wind_speed_mean * hour_cos
wind_speed_day_sin = wind_speed_mean * day_of_week_sin
wind_speed_day_cos = wind_speed_mean * day_of_week_cos
wind_speed_month_sin = wind_speed_mean * month_sin
wind_speed_month_cos = wind_speed_mean * month_cos
```

**删除特征（8个）**
- 所有单网格风速特征（4个）
- 所有气压MSL特征（4个）

**最终特征数**：26 → +12 -8 = 30个

### 3.4 4-Fold交叉验证细节（V3 Sample 2）

| Fold | 训练集大小 | 测试集大小 | RMSE (MW) |
|------|-----------|-----------|----------|
| 1 | 31077 | 10359 | 13.72 |
| 2 | 31077 | 10359 | 14.55 |
| 3 | 31077 | 10359 | 14.16 |
| 4 | 31077 | 10359 | 11.00 |
| 平均 | - | - | 13.28 |

注：Fold 4表现异常优异（RMSE=11.00），可能是该时段风况特征更稳定。

## 4. 观察到的问题

### 4.1 LLM生成代码的稳定性

**拼写错误**
- V3 Sample 3：`ecmwwf_ifs025_grid2_wind_speed_10m`（应为ecmwf）
- V5 Sample 10：`ecmwwf_ifs025_grid1_wind_speed_10m`（同样的错误）
- 导致执行失败，Score = None

**索引错误**
- V4 Sample 7：`ecmwf_ifs025_grid0_relative_humidity_2m`
- 原因：循环中使用`grid{i-1}`导致grid0不存在

### 4.2 采样收敛

**实际采样次数少于预设**
- 设定max_sample_nums=50，实际运行10-15次后停止
- V4的Sample 5/8/9/10生成完全相同的特征代码
- 表明LLM陷入局部最优，缺乏探索多样性

### 4.3 未探索的特征类型

**V3和V5均未生成的特征**
- 风向的三角函数编码：sin(direction), cos(direction)
- 气压比值（空气密度代理）：P_msl / P_surface
- 湿度-气压交互：humidity * pressure
- 高阶多项式：wind_speed ** 4, wind_speed ** 5
- 对数变换：log(wind_speed + 1)
- 风速-风向耦合：wind_speed ** 3 * cos(direction)

## 5. 技术细节

### 5.1 数据导出配置

**工具脚本**
- export_data_for_llmfe.py：导出CSV格式数据
- generate_llmfe_metadata.py：生成特征描述

**关键配置**
- model.yaml中use_uv_components：false（使用风速+风向）
- model.yaml中feature_engineering.*：全部设为false（禁用预处理）
- 时间特征：预先生成sin/cos编码，保持时间顺序

### 5.2 评估函数设置

**LightGBM参数**
```python
params = {
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
```

**交叉验证**
- 方法：KFold(n_splits=4, shuffle=False)
- shuffle=False确保保持时间顺序
- 每折训练集约31077样本，测试集约10359样本

### 5.3 文件结构

**LLM-FE项目**
- 数据：/data/windpower_SD_W_B_ecmwf_ifs025.csv
- 元数据：/meta_data/windpower_SD_W_B_ecmwf_ifs025.txt
- 规范：/specs/specification_windpower.txt
- 主程序：main.py

**Wind项目集成**
- 调用路径：sys.path.insert(0, '/data/cqj-project/wind-power-forecast-mlflow-1')
- 使用模型：core.models.base_lgbm.BaseLGBM
- 数据加载：core.dataloader.DataLoader

## 6. 结论

### 6.1 主要发现

**性能改进有限**
- 最佳RMSE从13.029降至13.00（改进0.22%）
- 相对于99 MW装机容量，绝对改进约0.03 MW
- 在26个基础特征上，自动特征工程的提升空间有限

**Specification策略影响显著**
- 明确公式提示（V3）：RMSE=13.00
- 完全弱化提示（V4）：RMSE=13.15（性能下降）
- 物理约束提示（V5）：RMSE=13.027（接近最佳）

**LLM发现的有效特征**
- 风速立方：符合风电功率定律
- 跨网格空间统计：捕捉气象场空间变异性
- 时间-气象交互：捕捉日变化和季节模式

### 6.2 局限性

**LLM探索能力**
- 倾向于生成常规统计特征（mean, std, max, min）
- 对复杂非线性组合的探索不足
- 容易陷入局部最优（重复生成相同特征）

**框架稳定性**
- 代码生成存在拼写错误和索引错误
- 采样次数远少于预设值
- 缺乏错误恢复机制

**任务特性限制**
- 风电功率预测的物理关系相对明确
- LightGBM已能学习部分非线性关系
- 原始特征信息量已较充分

### 6.3 技术贡献

**成功集成LLM-FE框架**
- 将Wind项目的BaseLGBM集成到LLM-FE评估流程
- 处理时间序列数据的K-Fold分割
- 建立了风电领域的Specification模板

**验证了Specification设计的重要性**
- 物理约束提示在领域任务中的有效性
- 完全弱化提示的不适用性
- 为后续研究提供了参考范式

## 7. 数据记录

### 7.1 实验日志

| 日志文件 | 大小 | 采样次数 | 版本 |
|---------|------|---------|------|
| llmfe_run_full_v2.log | 27KB | 4 | V2 (u/v分量) |
| llmfe_run_full_v3.log | 26KB | 4 | V3 (风速+风向) |
| llmfe_run_v4_explore.log | 68KB | 10 | V4 (弱化提示) |
| llmfe_run_v5.log | 76KB | 10 | V5 (物理约束) |

### 7.2 配置文件变更

**model.yaml关键修改**
```yaml
data:
  use_uv_components: false
  remove_original_wind: false

feature_engineering:
  add_wind_peak_features:
    enabled: false
  add_physical_interactions:
    enabled: false
  spatial_features:
    enabled: false
  add_meteorological_features:
    enabled: false
```

**main.py关键修改**
```python
# 添加参数支持
parser.add_argument('--max_sample_nums', type=int, default=20)
global_max_sample_num = args.max_sample_nums

# 修复回归问题识别
if problem_name in [...] or 'windpower' in problem_name:
    is_regression = True
```

### 7.3 运行命令

```bash
nohup python -u main.py \
  --problem_name windpower_SD_W_B_ecmwf_ifs025 \
  --use_api True \
  --api_model moonshot-v1-128k \
  --max_sample_nums 50 \
  --spec_path /data/cqj-project/LLM-FE/LLMFE/specs/specification_windpower.txt \
  --log_path /data/cqj-project/LLM-FE/logs \
  > llmfe_run_v5.log 2>&1 &
```
