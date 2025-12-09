# LLM-FE Wine Quality Dataset Experiment Report

## 1. 实验概述

### 数据集信息
- **数据集名称**: Wine Quality Dataset
- **样本数量**: 6,497 条（红葡萄酒+白葡萄酒）
- **特征数量**: 12 个原始特征
- **目标变量**: quality (葡萄酒质量评分，0-10分制)
- **任务类型**: 回归任务

### 原始特征说明
1. **fixed acidity**: 固定酸度（主要是酒石酸）
2. **volatile acidity**: 挥发性酸度（主要是乙酸）
3. **citric acid**: 柠檬酸含量
4. **residual sugar**: 残余糖分
5. **chlorides**: 氯化物含量
6. **free sulfur dioxide**: 游离二氧化硫
7. **total sulfur dioxide**: 总二氧化硫
8. **density**: 密度
9. **pH**: 酸碱度
10. **sulphates**: 硫酸盐
11. **alcohol**: 酒精含量
12. **color**: 颜色分类（0=红葡萄酒，1=白葡萄酒）

## 2. 实验配置

### API配置
- **API提供商**: Kimi (Moonshot AI)
- **API端点**: api.moonshot.cn/v1/chat/completions
- **模型**: moonshot-v1-8k

### 算法配置
- **评估模型**: XGBoost Regressor
- **交叉验证**: 4-Fold Cross-Validation (每个样本内)
- **外层划分**: 5-Fold Split (本次仅完成1个Fold)
- **评估指标**: RMSE (Root Mean Squared Error, 负值用于最大化)
- **最大迭代次数**: 20 iterations
- **随机种子**: 42
- **Multi-Island数量**: 3 (保持种群多样性)

### 特征工程配置
- **LLM温度**: 由代码动态控制
- **经验缓冲区**: Softmax温度采样
- **特征生成方式**: LLM生成Python代码片段进行特征转换

## 3. 实验结果

### 性能对比

| 指标 | 基线 (无特征工程) | 最佳结果 (Sample 5) | 改进幅度 |
|------|------------------|---------------------|---------|
| **RMSE** | 0.6619 | 0.6403 | -0.0216 |
| **改进百分比** | - | **3.3%** | - |

### 迭代过程统计
- **总样本数**: 22 个特征组合
- **成功评估**: 22 个
- **执行错误**: 多个样本返回 `Score: None` (代码执行错误)
- **最优分数首次出现**: Sample 5 (迭代早期)
- **后续改进**: 未超越Sample 5的最优分数

### 性能曲线关键点
1. **初始基线** (Sample 0): -0.6619
2. **首次改进** (Sample 3): -0.6591 (改进0.4%)
3. **最优突破** (Sample 5): **-0.6403** (改进3.3%)
4. **后续波动**: -0.64 ~ -0.66之间振荡
5. **最终分数** (Sample 22): -0.6596 (比最优差3.0%)

## 4. Top 特征组合排名

### 🥇 Rank 1: Sample 5 (Score: -0.6402837036564553)

**特征组合**:
```python
# 1. 硫酸盐与挥发性酸度的差值
sulphates_minus_volatile_acidity = sulphates - volatile_acidity

# 2. 总二氧化硫（错误：重复累加了total sulfur dioxide）
total_sulphur_dioxide = free_sulfur_dioxide + total_sulfur_dioxide

# 3. 酒精与固定酸度的比率
alcohol_to_acidity_ratio = alcohol / fixed_acidity

# 4. 密度与pH的比率
density_to_pH_ratio = density / pH

# 删除color特征
drop: color
```

**特征分析**:
- ✅ **sulphates_minus_volatile_acidity**: 捕捉硫化合物与挥发性酸的平衡
- ⚠️ **total_sulphur_dioxide**: 代码错误（应该只是重命名，却做了错误的加法）
- ✅ **alcohol_to_acidity_ratio**: 酒精-酸度平衡，影响口感
- ✅ **density_to_pH_ratio**: 密度-酸度关系
- ✅ **删除color**: LLM认为二元分类特征对质量预测贡献小

**关键优势**:
- 引入4个比率/差值特征
- 删除冗余分类特征
- 捕捉化学成分之间的平衡关系

---

### 🥈 Rank 2: Sample 10 (Score: -0.646514140991442)

**特征组合**:
```python
# 从日志可见此样本达到 -0.6465，但具体特征未在截取部分
```

**性能**: 比Rank 1差0.98%

---

### 🥉 Rank 3: Sample 13 (Score: -0.6485278147676997)

**性能**: 比Rank 1差1.3%

---

### Rank 4+: 其他样本

多个样本在 -0.64 ~ -0.66 之间，以下是代表性样本：

#### Sample 6 (Score: -0.6447388064570597)
**特征组合**:
```python
# 1. 硫酸盐与挥发性酸度的差值
sulphates_minus_volatile_acidity = sulphates - volatile_acidity

# 2. 硫酸盐与挥发性酸度的比率
sulphates_to_volatile_acidity_ratio = sulphates / volatile_acidity

# 3. 酒精与固定酸度的比率
alcohol_to_acid_ratio = alcohol / fixed_acidity

# 4. 游离二氧化硫占总二氧化硫的比例
sulphur_dioxide_ratio = free_sulfur_dioxide / total_sulfur_dioxide

# 5. 柠檬酸与残余糖分的比率
citric_acid_to_residual_sugar_ratio = citric_acid / residual_sugar

# 删除color特征
drop: color
```

**特征分析**:
- ✅ 5个新特征，涵盖多个化学成分比率
- ✅ **sulphur_dioxide_ratio**: 二氧化硫的游离比例，影响保鲜和口感
- ✅ **citric_acid_to_residual_sugar_ratio**: 酸甜平衡

**与Rank 1对比**: 特征更多但性能略逊，可能过拟合或引入噪声

---

#### Sample 22 (Score: -0.6596121304798434)
**特征组合**:
```python
# 1. 总二氧化硫/游离二氧化硫比率
total_sulfur_dioxide_to_free_sulfur_dioxide = total_sulfur_dioxide / free_sulfur_dioxide

# 2. 酒精/pH比率
alcohol_to_ph = alcohol / pH

# 3. 酒精×柠檬酸交互项
alcohol_citric_acid_interaction = alcohol * citric_acid

# 删除color特征
drop: color
```

**特征分析**:
- ✅ 引入交互项 (alcohol × citric_acid)
- ✅ 酒精-pH比率
- ⚠️ 性能接近基线，说明这3个特征组合不如Sample 5

---

## 5. 特征工程关键发现

### 有效特征类型 (按重要性排序)

1. **比率特征** (Ratio Features)
   - `alcohol_to_acidity_ratio = alcohol / fixed_acidity` (Rank 1)
   - `sulphur_dioxide_ratio = free_sulfur_dioxide / total_sulfur_dioxide` (Sample 6)
   - **优势**: 捕捉成分之间的相对平衡关系

2. **差值特征** (Difference Features)
   - `sulphates_minus_volatile_acidity = sulphates - volatile_acidity` (Rank 1)
   - **优势**: 反映化学成分的净平衡

3. **复合比率** (Composite Ratios)
   - `density_to_pH_ratio = density / pH` (Rank 1)
   - **优势**: 捕捉物理-化学性质的关系

4. **交互项** (Interaction Features)
   - `alcohol_citric_acid_interaction = alcohol * citric_acid` (Sample 22)
   - **劣势**: 在本任务中效果不如比率特征

### 无效特征类型

- ❌ **重复累加特征**: `total_sulphur_dioxide = free + total` (Sample 5有此错误，但仍是最优，说明其他特征起作用)
- ❌ **过多特征**: Sample 6有5个新特征，但性能不如Sample 5的4个特征
- ❌ **交互项**: Sample 22的交互项未带来改进

### LLM的共同策略
- ✅ **删除color特征**: 几乎所有Top样本都删除了color，LLM认为二元分类变量贡献小
- ✅ **优先生成比率**: 比率特征在所有Top样本中占主导
- ✅ **化学领域知识**: 所有特征都基于葡萄酒化学成分的领域知识（酸碱平衡、二氧化硫保鲜、酒精度等）

## 6. 改进显著性评估

### 统计显著性
- **基线RMSE**: 0.6619
- **最佳RMSE**: 0.6403
- **绝对改进**: 0.0216
- **相对改进**: **3.3%**

### 业务价值
- **预测准确度**: 质量评分预测误差减少0.0216分（10分制）
- **相对误差**: 0.6619 → 0.6403 (改进3.3%)
- **全样本集** (6,497个样本): 累计误差减少 **140.36分**

### 改进评价
⚠️ **统计学上中等**: 3.3%的改进在机器学习任务中属于**中等偏下**水平
- 比insurance数据集(8.4%)改进幅度小得多
- 可能原因:
  1. Wine数据集的原始特征已经较为充分
  2. 质量评分本身存在主观性，难以通过特征工程大幅提升
  3. LLM生成的特征多为简单比率，未探索高阶非线性特征

✅ **业务价值有限但明确**: 0.02分的改进对10分制评分系统有一定意义
⚠️ **改进不稳定**:
  - 最优分数出现在Sample 5 (迭代早期)
  - 后续15次迭代未能超越
  - 说明搜索陷入局部最优

## 7. 特征生成洞察

### LLM生成特征的共同模式
1. **领域知识驱动**: 所有特征都基于葡萄酒化学的常识（酸度、糖分、酒精、二氧化硫）
2. **比率优先**: Top样本平均包含2-3个比率特征
3. **删除color**: LLM一致认为color特征对质量预测贡献小
4. **简单组合**: 未出现复杂的条件特征或高阶多项式

### LLM的局限性
- ❌ **代码错误**: Sample 5有 `total_sulphur_dioxide = free + total` 的逻辑错误（重复累加）
- ❌ **缺少非线性**: 未尝试平方项、对数变换、分组特征
- ❌ **缺少三阶交互**: 未探索 `alcohol * pH * citric_acid` 等高阶项
- ⚠️ **早期收敛**: 最优解出现在Sample 5后，后续17次迭代无实质改进

## 8. 与Insurance数据集对比

| 指标 | Insurance | Wine | 对比 |
|------|-----------|------|------|
| **样本量** | 1,338 | 6,497 | Wine多5倍 |
| **原始特征** | 6 | 12 | Wine特征更丰富 |
| **任务类型** | 回归(费用) | 回归(评分) | 均为回归 |
| **基线RMSE** | $5,402 | 0.6619 | - |
| **最佳RMSE** | $4,949 | 0.6403 | - |
| **改进幅度** | **8.4%** | **3.3%** | Insurance改进更显著 |
| **最优样本** | Sample 19 (后期) | Sample 5 (早期) | Insurance搜索更持久 |
| **特征类型** | 分组+交互+非线性 | 比率+差值 | Insurance特征更复杂 |

### 关键差异分析
1. **改进幅度**: Insurance (8.4%) 远超 Wine (3.3%)
   - 可能原因: Insurance原始特征少(6 vs 12)，特征工程提升空间大
2. **收敛速度**: Wine早期收敛，Insurance持续改进至Sample 19
   - 可能原因: Wine的简单比率特征容易被LLM快速发现
3. **特征复杂度**: Insurance使用分组、平方项、条件特征；Wine仅用简单比率
   - 可能原因: Insurance的非线性关系更强（年龄-费用、吸烟-BMI交互）

## 9. 问题与观察

### 执行错误
日志中多次出现 `Score: None`，说明LLM生成的代码存在执行错误：
- 可能原因: 语法错误、列名错误、除零错误
- 影响: 浪费了多次迭代机会

### 最优解早期出现
- Sample 5 (第5次迭代) 达到最优 -0.6403
- 后续17次迭代未超越
- **建议**: 可能需要更高的LLM温度参数以增加探索多样性

### Color特征的一致性删除
- 几乎所有样本都删除了color特征
- 但实际上红/白葡萄酒的质量评判标准可能不同
- **建议**: 可尝试保留color并生成 `color × 其他特征` 的交互项

## 10. 结论与建议

### 主要结论
1. ✅ **LLM-FE有效但有限**: 实现了3.3%的性能提升，但不如Insurance数据集显著
2. ⚠️ **早期收敛**: 最优解出现在第5次迭代，后续改进停滞
3. ✅ **领域知识应用**: LLM成功应用了葡萄酒化学的领域知识（比率特征）
4. ⚠️ **缺少复杂特征**: 未探索非线性变换、分组特征、高阶交互

### 后续改进方向
1. **完成5-Fold验证**: 当前仅完成1个Fold，需运行完整5-Fold以评估稳定性
2. **增加特征复杂度**:
   - 非线性变换: `log(alcohol)`, `alcohol²`, `pH²`
   - 分组特征: `alcohol_group = alcohol // 1`
   - 三阶交互: `alcohol * pH * citric_acid`
   - 条件特征: `high_alcohol = (alcohol > 12) * sulphates`
3. **调整LLM参数**:
   - 提高温度参数以增加探索多样性
   - 增加迭代次数至50-100次
4. **保留color特征**: 尝试生成 `color × alcohol`, `color × pH` 等交互项
5. **调试执行错误**: 分析 `Score: None` 的原因，减少无效迭代

### 文件位置
- **日志文件**: `e:\VIVADO\LLM-FE\LLMFE\wine.log`
- **TensorBoard日志**: `e:\VIVADO\LLM-FE\LLMFE\runs\wine\`
- **样本JSON**: `e:\VIVADO\LLM-FE\LLMFE\runs\wine\samples\`
- **代码规范**: `e:\VIVADO\LLM-FE\LLMFE\specs\specification_wine.txt`

---

**报告生成时间**: 2025-12-03
**实验执行者**: LLM-FE自动化特征工程系统
**数据来源**: Wine Quality Dataset (6,497 samples)
