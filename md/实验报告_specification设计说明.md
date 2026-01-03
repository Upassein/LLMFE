## 4. Specification设计方法

### 4.1 设计思路

**核心问题：如何引导LLM发现有效的风电特征？**

**原始LLM-FE框架的设计哲学：**
- 极简specification：只有任务描述 + 一个初始示例
- 依赖Experience Buffer的动态学习
- 让LLM通过遗传算法自由探索

例如Insurance的specification：
```python
"""
Task: Estimate medical cost billed by health insurance
Thought 1: Combining age and BMI may provide insight into health risk
New Feature 1: age_bmi = (age * bmi)
"""
```

**风电任务的挑战：**
- 特征复杂：26个气象特征 vs Insurance的6个
- 领域门槛：需要物理知识（功率定律、空气密度）
- 探索空间大：多网格 × 多气象要素 × 时间
- 实验教训：V2中GFS删除所有风速导致性能崩溃

**我的改进策略（区别于原始框架）：**
1. **主动引导** vs 自由探索：提供结构化特征建议
2. **明确约束** vs 开放探索：禁止删除、限制数量
3. **分类组织** vs 单一示例：BASIC/INTERACTIONS/SPATIAL
4. **简化知识** vs 无领域知识："speed^3"等物理直觉
5. **历史反馈** vs 纯遗传：告知baseline和最佳性能

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

**第四部分：分类的特征建议（核心创新）**
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

### 4.3 与原始框架的对比

| 维度 | 原始LLM-FE | 我的风电Specification |
|------|-----------|---------------------|
| **Specification长度** | 极简（3行） | 详细（80行） |
| **特征删除策略** | 未限制 | "DO NOT delete" |
| **示例代码** | 1个初始示例 | 每类2-3个具体公式 |
| **领域知识** | 无 | 简化物理直觉（speed^3） |
| **复杂度控制** | 无 | "5-10个新特征/次" |
| **特征分类** | 无 | BASIC/INTERACTIONS/SPATIAL |
| **历史反馈** | 无 | "Baseline 13.14, Best 12.94" |
| **设计哲学** | 依赖遗传算法自由探索 | 主动引导LLM探索方向 |

**关键区别：**
- 原始框架：适合简单数据集（6特征），依赖框架的Experience Buffer机制
- 我的改进：适合复杂领域任务（26特征+物理知识），通过详细提示词主动引导

### 4.4 设计效果验证

**成功之处：**
- ECMWF V6：13.14 → 12.94 MW（4次迭代，1.5%改进）
- 优化后的specification更结构化，便于复制到ICON/GFS

**待改进：**
- 目前仍只跑4次就停止（虽设置max_sample_nums=20）
- 需进一步研究如何提高LLM探索多样性
- 考虑动态调整提示词复杂度
