# LLM-FE在风电功率预测中的应用实验报告

## 1. 实验配置

**数据集**

- 风场：SD_W_B（装机99 MW），41436样本
- 气象源：ECMWF IFS025（26特征：20气象+6时间）
- 评估：4-Fold时间序列CV，RMSE指标

**框架**

- LLM：Moonshot-v1-128k
- 模型：LightGBM (max_depth=8, lr=0.03)
- 算法：Multi-Island遗传算法 + Experience Buffer

## 2. 实验结果

### 2.1 性能对比

| 版本     | Specification策略 | 采样次数 | 最佳RMSE (MW) | 改进   |
| -------- | ----------------- | -------- | ------------- | ------ |
| Baseline | 无特征工程        | -        | 13.029        | -      |
| V3       | 明确公式提示      | 4        | 13.00         | 0.22%  |
| V4       | 完全弱化提示      | 10       | 13.15         | -0.92% |
| V5       | 物理约束提示      | 10       | 13.027        | 0.15%  |

### 2.2 最佳特征（V3）

**新增特征（15个）**

```python
# 风速立方（符合功率定律）
wind_power_potential_grid{1-4} = wind_speed_{1-4} ** 3

# 气压差（空气密度指示）
pressure_diff_grid{1-4} = pressure_msl_{1-4} - surface_pressure_{1-4}

# 空间梯度（跨网格差异）
rh_diff_grid12/23/34 = humidity_{i+1} - humidity_{i}
wind_dir_diff_grid12/23/34 = direction_{i+1} - direction_{i}
```

**删除特征（8个）**

- 所有风向特征（4个）
- 所有地面气压特征（4个）

最终：26 → 33个特征

## 3. 关键发现

### 3.1 Specification设计的重要性

**明确公式提示（V3）效果最好：**

- 具体建议特征类型：wind_speed^3, sin/cos(direction), 压力梯度
- 明确跨网格操作：mean, std, max, min
- 提供历史反馈：RMSE范围13.0-13.3 MW

**完全弱化提示（V4）效果最差：**

- 只有抽象领域知识，无具体公式
- LLM生成保守，缺乏有效探索
- RMSE反而上升0.92%

### 3.2 LLM发现的有效模式

**符合物理规律的特征：**

- 风速立方：P ∝ v³（风电功率定律）
- 气压差：反映空气密度变化

**数据驱动的特征：**

- 跨网格空间差异：捕捉气象场变异性
- 时间-气象交互：捕捉日变化模式

### 3.3 局限性

**LLM探索能力：**

- 倾向于常规统计特征，缺乏创造性组合
- 容易陷入局部最优（V4的Sample 5/8/9/10完全相同）
- 代码生成存在拼写错误（ecmwwf）和索引错误（grid0）

**改进空间有限：**

- 最佳改进仅0.22%（RMSE 13.029→13.00）
- 风电预测物理关系明确，LightGBM已能学习主要非线性
- 原始26特征信息量较充分

**未探索的特征：**

- 风向三角编码：sin(direction), cos(direction)
- 气压比值：P_msl / P_surface
- 风速-风向耦合：wind_speed³ * cos(direction)

## 4. Specification设计方法

### 4.1 设计思路

**核心问题：如何引导LLM发现有效的风电特征？**

**原始LLM-FE框架的设计哲学：**- 极简specification：只有任务描述 + 一个初始示例- 依赖Experience Buffer的动态学习- 让LLM通过遗传算法自由探索例如Insurance的specification：

```python
"""Task: Estimate medical cost billed by health insuranceThought 1: Combining age and BMI may provide insightNew Feature 1: age_bmi = (age * bmi)"""
```

**风电任务的挑战：**

- 特征复杂：26个气象特征 vs Insurance的6个
- 领域门槛：需要物理知识（功率定律、空气密度）
- 探索空间大：多网格 × 多气象要素 × 时间
- 实验教训：V2中GFS删除所有风速导致性能崩溃

**我的改进策略（区别于原始框架）：**

1. **主动引导** vs 自由探索：提供结构化特征建议
2. **明确约束** vs 开放探索：禁止删除、限制数量
3. **分类组织** vs 单一示例：BASIC/INTERACTIONS/SPATIAL4.
   **简化知识** vs 无领域知识："speed^3"等物理直觉5.
   **历史反馈** vs 纯遗传：告知baseline和最佳性能

### 4.2 Specification结构

**第一部分：特征列表 + 任务描述**

```
Available features (26 features total):
- Wind speed and direction at 10m for 4 spatial grids
- Relative humidity at 2m for 4 grids
- Pressure (MSL and surface) for 4 grids
- Time features: hour_sin/cos, day_of_week_sin/cos, month_sin/cos

Task: Predict wind power generation (range: 0-99 MW, mean: ~20 MW)
```

**第二部分：CRITICAL RULES（最重要）**

```
1. DO NOT delete or remove any original features - only ADD new features
2. Focus on simple, interpretable transformations
3. Start with basic features before trying complex combinations
```

**第三部分：简化的领域知识**

```
Helpful domain knowledge (simplified):
- Wind power increases with wind speed (roughly proportional to speed^3)
- Different wind directions have different power outputs
- Atmospheric pressure affects air density
- Spatial differences across grids indicate wind patterns
```

**第四部分：分类的特征建议**

```
BASIC TRANSFORMATIONS:
- Wind speed powers: speed^2, speed^3 (wind power law)
- Direction encoding: sin(direction), cos(direction)
- Pressure ratio: pressure_msl / surface_pressure

INTERACTIONS:
- Speed × direction: speed * sin(direction)
- Speed × time: speed * hour_sin (diurnal patterns)
- Speed × humidity: captures air density effect

SPATIAL FEATURES:
- Average wind speed across grids: mean(speed_grid1, ...)
- Wind speed variance: std across grids (turbulence indicator)
- Pressure differences: pressure_grid1 - pressure_grid2

BINNING:
- Wind speed categories: speed_category = speed // 2
- Hour groups: hour_period = hour // 6
```

**第五部分：约束和反馈**

```
IMPORTANT:
- Keep all original 26 features unchanged
- Add only 5-10 new features per iteration
- Avoid overly complex transformations
- Test simple ideas first: speed^2 and speed^3 are often most useful

Previous results:
- Baseline RMSE: 13.14 MW
- Best with basic polynomials: 12.94 MW (1.5% improvement)
```

### 4.3 与V6版本的对比

| 维度                 | V6 (旧版)                     | 当前优化版                   |
| -------------------- | ----------------------------- | ---------------------------- |
| **特征删除**   | "Remove at most 40%"          | "DO NOT delete"              |
| **示例代码**   | 无                            | 每类提供2-3个具体公式        |
| **领域知识**   | "the relationship is complex" | "speed^3" 简化直观           |
| **复杂度控制** | 无明确限制                    | "5-10个新特征/次"            |
| **特征分类**   | "Think creatively about..."   | BASIC/INTERACTIONS/SPATIAL   |
| **历史反馈**   | 无                            | "Baseline 13.14, Best 12.94" |

### 4.4 设计效果验证

**成功之处：**

- ECMWF V6（旧spec）：13.14 → 12.94 MW（4次迭代）
- 优化后的ICON/GFS spec虽因dart+max_depth=60卡住，但提示词设计更科学

**待改进：**

- 目前仍只跑4次就停止（虽设置max_sample_nums=20）
- 需进一步研究如何提高LLM探索多样性

## 6. 技术实现

### 4.1 数据准备

**配置修改（model.yaml）：**

```yaml
data:
  use_uv_components: false  # 使用风速+风向而非u/v分量

feature_engineering:
  *: enabled: false  # 禁用所有预处理，让LLM-FE从原始数据出发
```

### 4.2 关键代码修改

**main.py：**

```python
parser.add_argument('--max_sample_nums', type=int, default=20)
if 'windpower' in problem_name:
    is_regression = True
```

**specification_windpower.txt：**

- 集成Wind项目的BaseLGBM评估器
- 时间序列4-Fold CV（shuffle=False）
- 物理约束提示词

### 4.3 运行命令

```bash
nohup python -u main.py \
  --problem_name windpower_SD_W_B_ecmwf_ifs025 \
  --use_api True \
  --api_model moonshot-v1-128k \
  --max_sample_nums 50 \
  --spec_path specs/specification_windpower.txt \
  > llmfe_run.log 2>&1 &
```

## 6. 结论

**主要贡献：**

- 成功将LLM-FE集成到风电预测任务
- 验证了Specification设计对性能的决定性影响
  **实际价值：**
- 性能改进微小（0.22%），不足以证明LLM-FE在该任务的优势
- LLM主要发现了已知的物理规律（风速立方）
- 对于物理规律明确的任务，专家特征工程可能更高效

**未来方向：**

- 优化Specification提示词（借鉴Insurance数据集成功经验）
- 改进采样策略以增加探索多样性
