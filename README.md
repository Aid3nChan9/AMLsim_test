# 基于随机森林的可疑洗钱行为识别

## 项目简介

本项目构建了一套完整的金融交易反洗钱（AML）检测模型，经历了 **11 次迭代优化**。最终版 `aml_model_v11.py` 采用**集成学习（随机森林 + XGBoost 软投票）** 结合**复杂网络拓扑特征**与**时序行为特征**，通过 **SMOTETomek 混合采样**处理数据不平衡，并以 **F1.5 最大化策略**优化决策阈值，实现对可疑洗钱账户的精准识别。

**最终版模型**：`aml_model_v11.py`

## 模型核心特性

### 1. 多维度特征工程

| 特征类别 | 具体特征 | 数量 |
|---------|---------|:--:|
| 基础属性 | `INIT_BALANCE` | 1 |
| 交易统计（发送/接收） | `send_sum`, `send_mean`, `send_max`, `send_std`, `send_count` / `recv_*` | 10 |
| 衍生特征 | `out_in_ratio`, `net_flow` | 2 |
| 网络拓扑 | `out_degree`, `in_degree`, `total_degree`, `pagerank`, `clustering` | 5 |
| 时序特征 | `active_days_send/recv`, `daily_tx_send/recv`, `night_ratio_send/recv`, `max_daily_amount_send/recv`, `max_daily_count_send/recv` | 10 |
| 类别编码 | `country_freq`（频数编码）, `type_*`（独热编码）, `behavior_freq`（频数编码） | 动态 |

### 2. 严格防数据穿越的数据划分

- **时间序列划分**：按交易时间戳排序后，前 80% 为训练+验证期，后 20% 为测试期
- **验证集**：训练+验证期内再按时间取后 10%
- **图构建隔离**：网络拓扑特征**仅基于训练+验证期交易**构建，测试集复用该图结构，杜绝未来信息泄露
- **训练/验证划分**：在特征矩阵层面按账户 ID 随机分层划分（80%/20%）

### 3. 不平衡处理

- **混合采样**：`SMOTETomek`（SMOTE 过采样 + Tomek Links 欠采样，清理边界噪声）
- **代价敏感学习**：`class_weight='balanced_subsample'`

### 4. 集成学习

| 基学习器 | 参数 | 权重（默认） |
|---------|------|:--:|
| 随机森林 | `n_estimators=200`, `max_depth=None`, `min_samples_split=2` | 0.5 |
| XGBoost | `n_estimators=200`, `max_depth=6`, `learning_rate=0.05` | 0.5 |

- 支持 **GPU 加速**（cuML 随机森林，自动检测回退 CPU）
- 集成方式：**概率加权平均**（可配置权重 `--rf_weight` / `--xgb_weight`）

### 5. 阈值优化策略

**F1.5 最大化**：在验证集上搜索使 F1.5 分数最大的决策阈值。F1.5 对召回率的权重是精确率的 1.5² = 2.25 倍，兼顾反洗钱场景对召回率的高要求。

```
搜索范围: 0.01 ~ 0.99，步长 200
目标:     max F1.5
```

### 6. 模型可解释性（SHAP）

- **全局解释**：SHAP Summary Plot（特征贡献排序）
- **局部解释**：SHAP Waterfall Plot（单个真实可疑账户的决策推理解释）

## 环境要求

- Python ≥ 3.8
- **可选**：NVIDIA GPU + cuML（用于 GPU 加速随机森林）

| 包名 | 用途 |
|------|------|
| `pandas` | 数据处理 |
| `numpy` | 数值计算 |
| `networkx` | 交易网络图构建与分析 |
| `scikit-learn` | 随机森林、评估指标 |
| `imbalanced-learn` | SMOTETomek 混合采样 |
| `xgboost` | XGBoost 集成学习 |
| `matplotlib` | 基础绘图 |
| `seaborn` | 混淆矩阵热力图 |
| `shap` | 模型可解释性分析 |

安装依赖：

```bash
pip install pandas numpy networkx scikit-learn imbalanced-learn xgboost matplotlib seaborn shap

# 可选：GPU 加速
pip install cuml-cu12  # 根据 CUDA 版本选择
```

## 数据集

使用 AMLSim 生成的仿真交易数据，需包含以下两个 CSV 文件：

### transactions.csv（交易数据）

| 字段 | 说明 |
|------|------|
| `SENDER_ACCOUNT_ID` | 发送方账户ID |
| `RECEIVER_ACCOUNT_ID` | 接收方账户ID |
| `TX_AMOUNT` | 交易金额 |
| `TX_TYPE` | 交易类型 |
| `TIMESTAMP` | 交易时间戳（整数步长） |

### accounts.csv（账户数据）

| 字段 | 说明 |
|------|------|
| `ACCOUNT_ID` | 账户ID |
| `IS_FRAUD` | 是否可疑（1=可疑, 0=正常） |
| `INIT_BALANCE` | 初始余额 |
| `COUNTRY` | 所属国家 |
| `ACCOUNT_TYPE` | 账户类型 |
| `TX_BEHAVIOR_ID` | 交易行为模式ID |

数据目录结构示例：

```
data/10Kvertices-1Medges/
├── transactions.csv
└── accounts.csv
```

## 运行方法

```bash
python aml_model_v11.py --data_dir "数据集文件夹路径" [--rf_weight 0.5] [--xgb_weight 0.5]
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|:--:|------|------|
| `--data_dir` | ✓ | — | 存放 `transactions.csv` 和 `accounts.csv` 的目录路径 |
| `--rf_weight` | | `0.5` | 随机森林权重（0~1），自动归一化 |
| `--xgb_weight` | | `0.5` | XGBoost 权重（0~1），自动归一化 |

### 运行示例

```bash
# 默认等权集成
python aml_model_v11.py --data_dir "./data/10Kvertices-1Medges"

# 随机森林主导（RF 80%, XGBoost 20%）
python aml_model_v11.py --data_dir "./data/10Kvertices-1Medges" --rf_weight 0.8 --xgb_weight 0.2

# 纯 XGBoost
python aml_model_v11.py --data_dir "./data/10Kvertices-1Medges" --rf_weight 0 --xgb_weight 1.0

# 纯随机森林
python aml_model_v11.py --data_dir "./data/10Kvertices-1Medges" --rf_weight 1.0 --xgb_weight 0
```

## 输出结果

运行完成后，在脚本同级目录下生成以下文件：

| 文件名 | 说明 |
|------|------|
| `confusion_matrix_v11.png` | 测试集混淆矩阵（含阈值与总样本数） |
| `feature_importance_v11.png` | 特征重要性排序条形图 |
| `shap_summary_v11.png` | SHAP 全局特征贡献汇总图 |
| `shap_waterfall_v11.png` | 单个真实可疑账户的 SHAP 瀑布图 |
| `suspicious_accounts_v11.csv` | 模型预测为可疑的账户名单（按风险概率降序） |

### suspicious_accounts_v11.csv 字段

| 字段 | 说明 |
|------|------|
| `ACCOUNT_ID` | 账户ID |
| `FRAUD_PROBABILITY` | 模型预测的可疑概率 |
| `PREDICTION` | 预测标签（1=可疑, 0=正常） |
| `TRUE_LABEL` | 真实标签（1=可疑, 0=正常） |

## 模型处理流程

```
加载数据 ──→ 按时间划分（80/10/10）
                │
    ┌───────────┴───────────┐
    ▼                       ▼
训练+验证期                 测试期
    │                       │
特征工程                    独立构建特征
├─ 统计特征                 （复用训练图计算网络特征）
├─ 网络特征（仅用训练+验证交易建图）
├─ 时序特征
└─ 类别编码
    │                       │
    ▼                       ▼
训练/验证随机分层           模型预测 ──→ 阈值判定
    │
SMOTETomek 混合采样
    │
RF + XGBoost 训练
    │
验证集 F1.5 阈值搜索
    │
    └──→ 测试集评估 ──→ 输出图表 + 可疑名单
```

## 版本演进总览

| 版本 | 核心变化 | 数据划分 | 采样 | 模型 | 阈值 | 特征 |
|------|---------|:--:|:--:|:--:|:--:|:--:|
| v1 | 基础随机森林 | 随机 70/30 | 无 | RF+GridSearchCV | 0.5 | 20维（统计+网络+LabelEncode） |
| v2 | +BorderlineSMOTE +验证集 | 随机 60/20/20 | BorderlineSMOTE | RF+GridSearchCV | F1搜索 | 同v1 |
| v3 | 模块化 + argparse | 同v2 | 同v2 | 同v2 | 同v2 | 同v2 |
| v4 | +高级特征 +双策略 | 同v2 | 同v2 | 同v2 | recall/fbeta | **+2 = 22维** |
| v5 | 精简单策略 +可疑名单 | 同v2 | 同v2 | 同v2 | recall only | 同v4 |
| v6 | 恢复双策略 | 同v2 | 同v2 | 同v2 | recall+fbeta | 同v4 |
| **v7** | **时间划分 +时序特征** | **时间 80/10/10** | BorderlineSMOTE | RF(cuML可选)+GridSearchCV | recall | **+10时序 -1路径 = 31维** |
| **v8** | **SMOTETomek +XGBoost集成** | 时间 80/10/10 | **SMOTETomek** | **RF + XGBoost** | **F0.5** | 31+2高级(F0占位) |
| v9 | F2.0+recall底线 | 同v8 | 同v8 | 同v8 | F2.0(recall≥0.85) | 同v8 |
| v10 | 去高级特征 简化为recall | 同v8 | 同v8 | 同v8 | recall | **~40维**（无高级特征占位） |
| **v11** ⭐ | **F1.5平衡 +去后处理** | 时间 80/10/10 | SMOTETomek | RF + XGBoost | **F1.5** | **~40维**（时序+网络+统计） |

### 关键里程碑详解

| 里程碑 | 版本 | 意义 |
|--------|:--:|------|
| **数据穿越修复** | v7 | 从随机划分改为按时间序列划分，网络图仅用历史交易构建，杜绝未来信息泄露 |
| **时序特征引入** | v7 | 新增活跃天数、日均交易、深夜交易比例、单日最大交易等 10 维时序行为特征 |
| **集成学习** | v8 | 随机森林 + XGBoost 概率加权投票，提升模型鲁棒性 |
| **混合采样** | v8 | SMOTETomek 替换 BorderlineSMOTE，过采样同时清理边界噪声 |
| **GPU 加速** | v7 | 可选 cuML 随机森林，训练速度提升数倍 |
| **F1.5 平衡策略** | v11 | 折中 F0.5（偏精确率）与 F2.0（偏召回率），更稳健 |
| **去后处理** | v11 | 删除静默账户强制排除逻辑，让模型基于图特征自行判断 |

### v8-v11 核心差异（选择 v11 的理由）

| 对比维度 | v8 | v9 | v10 | **v11** |
|---------|:--:|:--:|:--:|:--:|
| 阈值策略 | F0.5（精确率优先） | F2.0+底线（召回率优先） | recall 80%（召回率优先） | **F1.5（平衡）** |
| 高级特征 | behavior_deviation + penetration_rate | 同v8 | **无（移除）** | **无** |
| 静默账户后处理 | 有 | 有 | 有 | **无** |
| 集成权重 | 0.3/0.7（偏XGBoost） | 0.3/0.7（偏XGBoost） | 0.5/0.5 | **0.5/0.5** |
| 特征空间一致性 | ❌（训练集占位值为0） | ❌ | ✅ | ✅ |

**v11 的设计哲学**：在确定高级特征无法安全融入训练集（存在数据穿越风险和时间不一致）后，果断移除；同时移除启发式后处理规则，让端到端模型自行学习。F1.5 阈值在反洗钱场景的"宁可错杀也不漏杀"与"控制误报成本"之间取得平衡。

## 引用

如需使用本项目，请引用本毕业论文成果。

---

*毕业设计：基于集成学习的反洗钱可疑交易识别研究*
