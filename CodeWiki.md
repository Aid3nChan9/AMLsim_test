# AMLsim_test 项目 CodeWiki (v11.0 Final)

## 1. 项目概述

### 1.1 项目目标
本项目是一个**反洗钱(AML)检测系统**，基于 AMLsim 生成的仿真金融交易数据，利用**集成学习（随机森林 + XGBoost）**结合**复杂网络拓扑特征**与**时序行为特征**，通过 **SMOTETomek 混合采样**处理数据不平衡，并以 **F1.5 最大化策略**优化决策阈值，实现对可疑洗钱账户的精准识别。

### 1.2 应用场景
- **金融风控**：帮助银行或金融机构自动识别潜在的洗钱账户
- **学术研究**：毕业论文实验代码，展示反洗钱检测的完整迭代流程（v1 → v11）
- **模型可解释性**：使用 SHAP 值解释模型预测结果，满足金融监管的可解释性要求

### 1.3 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 数据处理 | pandas, numpy | 数据加载、聚合、特征构建 |
| 网络分析 | networkx | 交易有向图构建、出入度、PageRank、聚集系数 |
| 机器学习 | scikit-learn, xgboost | 随机森林 + XGBoost 集成学习 |
| 不平衡处理 | imbalanced-learn | SMOTETomek 混合采样 |
| GPU 加速 | cuML (可选) | 随机森林 GPU 训练 |
| 可视化 | matplotlib, seaborn | 混淆矩阵、特征重要性 |
| 可解释性 | shap | SHAP Summary + Waterfall |
| 前端界面 | gradio | Web 端可视化交互（aml_ui.py） |

---

## 2. 项目结构

```
AMLsim_test/
├── aml_model.py           # v1.0：基础随机森林
├── aml_model_v2.py        # v2.0：+BorderlineSMOTE + 验证集
├── aml_model_v3.py        # v3.0：模块化 + argparse
├── aml_model_v4.py        # v4.0：+高级特征 + 双策略
├── aml_model_v5.py        # v5.0：精简单策略 + 名单输出
├── aml_model_v6.py        # v6.0：恢复双策略
├── aml_model_v7.py        # v7.0：时间划分 + 时序特征 + GPU
├── aml_model_v8.py        # v8.0：SMOTETomek + XGBoost集成
├── aml_model_v9.py        # v9.0：F2.0 + 召回底线
├── aml_model_v10.py       # v10.0：去高级特征 + recall
├── aml_model_v11.py       # v11.0 ⭐ 最终版
├── aml_ui.py              # Gradio Web 交互界面
├── README.md              # 项目文档
├── requirements_ui.txt    # Python 依赖清单
├── CodeWiki.md            # 本文档
└── .gitignore             # Git 忽略配置
```

---

## 3. v11.0 核心算法详解

### 3.1 特征工程（~40维）

#### 3.1.1 交易统计特征（12维）
按账户聚合发送/接收统计：
- `send_sum`, `send_mean`, `send_max`, `send_std`, `send_count` (5维)
- `recv_sum`, `recv_mean`, `recv_max`, `recv_std`, `recv_count` (5维)
- `out_in_ratio`（收发笔数比）、`net_flow`（净流量） (2维)

#### 3.1.2 网络拓扑特征（5维）
仅基于训练+验证期交易构建有向图：
- `out_degree`：出度
- `in_degree`：入度
- `total_degree`：总度
- `pagerank`：PageRank 值
- `clustering`：聚集系数

#### 3.1.3 时序行为特征（10维）

| 发送方 | 接收方 | 说明 |
|--------|--------|------|
| `active_days_send` | `active_days_recv` | 活跃天数 |
| `daily_tx_send` | `daily_tx_recv` | 日均交易笔数 |
| `night_ratio_send` | `night_ratio_recv` | 深夜（0-6点）交易金额占比 |
| `max_daily_amount_send` | `max_daily_amount_recv` | 单日最大交易金额 |
| `max_daily_count_send` | `max_daily_count_recv` | 单日最大交易笔数 |

#### 3.1.4 类别编码（动态维度）
- `COUNTRY`：频数编码 → `country_freq`
- `ACCOUNT_TYPE`：独热编码 → `type_*`
- `TX_BEHAVIOR_ID`：频数编码 → `behavior_freq`

### 3.2 严格防数据穿越

```
时间轴
├─ 80% ────────────────────────────┤─ 10% ─┤─ 10% ─┤
│        训练+验证期                 │ 验证   │ 测试   │
│   ┌─ 72% ──┤─ 8% ─┐              │        │        │
│   │ 训练     │验证   │              │        │        │
│   └──────────┴───────┘              │        │        │
│   图构建仅基于训练+验证交易           │        │        │
│                                     │        │        │
└─────────────────────────────────────┴────────┴────────┘
```

- 网络图**仅基于训练+验证期交易**构建，测试集复用
- 时序特征各时期独立计算
- 杜绝未来信息泄露到训练中

### 3.3 SMOTETomek 混合采样
- **SMOTE**：合成少数类过采样
- **Tomek Links**：清理边界重叠样本，降低噪声
- 仅作用于训练集

### 3.4 集成学习

| 基学习器 | 关键参数 | 默认权重 |
|---------|---------|:--:|
| 随机森林 | `n_estimators=200`, `max_depth=None` | 0.5 |
| XGBoost | `n_estimators=200`, `max_depth=6`, `lr=0.05` | 0.5 |

集成方式：概率加权平均（支持 GPU 加速）

### 3.5 F1.5 阈值优化

```
搜索范围: 0.01 ~ 0.99，步长 200
目标:     max F1.5 (召回率权重 = 精确率的 2.25 倍)
```

F1.5 在反洗钱场景的"宁可错杀也不漏杀"与"控制误报成本"之间取得平衡。

---

## 4. 版本演进关键里程碑

| 里程碑 | 版本 | 意义 |
|--------|:--:|------|
| 网络拓扑特征引入 | v2 | 出入度、PageRank、聚集系数 |
| Borderline-SMOTE | v2 | 首次处理样本不平衡 |
| 高级特征 | v4 | 资金周转率 + 平均最短路径 |
| 数据穿越修复 | v7 | 时间序列划分替代随机划分 |
| 时序特征 | v7 | 10维时序行为特征 |
| GPU 加速 | v7 | cuML 可选 |
| SMOTETomek | v8 | 混合采样替代 Borderline-SMOTE |
| 集成学习 | v8 | RF + XGBoost 软投票 |
| F1.5 平衡 | v11 | 折中精确率与召回率 |
| 去后处理 | v11 | 端到端模型，无启发式规则 |

---

## 5. 使用指南

### 5.1 环境安装
```bash
pip install -r requirements_ui.txt
```

### 5.2 模型训练
```bash
python aml_model_v11.py --data_dir <数据文件夹路径>

# 自定义集成权重
python aml_model_v11.py --data_dir <路径> --rf_weight 0.8 --xgb_weight 0.2
```

### 5.3 Gradio 交互界面
```bash
python aml_ui.py
```
访问 `http://localhost:7862` 使用 Web 界面。

---

## 6. 输出文件

| 文件 | 说明 |
|------|------|
| `confusion_matrix_v11.png` | 混淆矩阵（含阈值 + 样本数） |
| `feature_importance_v11.png` | 特征重要性排序 |
| `shap_summary_v11.png` | SHAP 全局汇总 |
| `shap_waterfall_v11.png` | SHAP 单样本瀑布图 |
| `suspicious_accounts_v11.csv` | 可疑账户名单（按概率降序） |

---

*文档更新时间：2026-05-23*
*项目作者：张清祥（毕业论文项目）*
