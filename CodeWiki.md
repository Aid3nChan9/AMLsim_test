# AMLsim_test 项目 CodeWiki

## 1. 项目概述

### 1.1 项目目标
本项目是一个**反洗钱(AML)检测系统**，基于 AMLsim 生成的合成金融交易数据，利用机器学习技术识别可疑账户。项目通过分析账户的交易行为特征和网络拓扑特征，构建分类模型来预测账户是否为欺诈账户。

### 1.2 应用场景
- **金融风控**：帮助银行或金融机构自动识别潜在的洗钱账户
- **学术研究**：作为毕业论文的实验代码，展示反洗钱检测的完整流程
- **模型可解释性**：使用 SHAP 值解释模型预测结果，满足金融监管的可解释性要求

### 1.3 技术栈
- **数据处理**：pandas, numpy
- **网络分析**：networkx
- **机器学习**：scikit-learn (RandomForestClassifier, GridSearchCV)
- **可视化**：matplotlib, seaborn
- **可解释性**：shap

---

## 2. 项目结构

```
AMLsim_test/
├── aml_model.py          # 主程序：完整的AML检测流程
├── untitle1.py           # 辅助脚本：查看交易数据列名
├── untitle2.py           # 辅助脚本：查看数据文件列表和基础统计
├── untitle3.py           # 辅助脚本：查看账户数据分布
├── confusion_matrix.png  # 输出：混淆矩阵可视化
├── feature_importance.png # 输出：特征重要性图
├── shap_summary.png      # 输出：SHAP特征重要性汇总图
├── shap_waterfall.png    # 输出：单个样本的SHAP瀑布图
└── .gitignore            # Git忽略配置
```

### 2.1 文件职责

| 文件 | 类型 | 职责 |
|------|------|------|
| `aml_model.py` | 主程序 | 完整的数据加载、特征工程、模型训练、评估和可视化流程 |
| `untitle1.py` | 辅助脚本 | 快速查看 transactions.csv 的列结构和样本数据 |
| `untitle2.py` | 辅助脚本 | 列出数据目录文件，查看交易数据统计信息 |
| `untitle3.py` | 辅助脚本 | 查看 accounts.csv 中可疑账户的分布情况 |

---

## 3. 核心算法详解

### 3.1 特征工程

#### 3.1.1 交易行为统计特征（10维）
从 `transactions.csv` 按账户聚合计算：

**发送方统计（5维）**：
- `send_sum`: 发送金额总和
- `send_mean`: 发送金额平均值
- `send_max`: 发送金额最大值
- `send_std`: 发送金额标准差
- `send_count`: 发送交易次数

**接收方统计（5维）**：
- `recv_sum`: 接收金额总和
- `recv_mean`: 接收金额平均值
- `recv_max`: 接收金额最大值
- `recv_std`: 接收金额标准差
- `recv_count`: 接收交易次数

**衍生特征（2维）**：
- `out_in_ratio`: 出入度比（发送次数/接收次数），识别单向资金流动模式
- `net_flow`: 净流量（发送总额 - 接收总额），识别资金净流入/流出账户

#### 3.1.2 网络拓扑特征（5维）
使用 NetworkX 构建交易网络（有向图）：
- `out_degree`: 出度（该账户作为发送方的交易对手数量）
- `in_degree`: 入度（该账户作为接收方的交易对手数量）
- `total_degree`: 总度（出入度之和）
- `pagerank`: PageRank值，衡量账户在网络中的重要性
- `clustering`: 聚集系数，衡量账户邻居之间的连接紧密程度

#### 3.1.3 类别特征编码（3维）
- `country_enc`: 账户所属国家编码
- `type_enc`: 账户类型编码
- `behavior_enc`: 交易行为ID编码

**总计：18维特征**

### 3.2 机器学习模型

#### 3.2.1 模型选择
- **算法**：随机森林（Random Forest）
- **原因**：
  - 处理高维特征能力强
  - 天然支持特征重要性评估
  - 对类别不平衡数据鲁棒性好
  - 支持 SHAP 可解释性分析

#### 3.2.2 超参数调优
使用 `GridSearchCV` 进行网格搜索：
```python
param_grid = {
    'n_estimators': [100, 200],      # 树的数量
    'max_depth': [10, 20, None],     # 最大深度
    'min_samples_split': [2, 5],     # 内部节点再划分所需最小样本数
}
```
- **交叉验证**：3折
- **评估指标**：F1-score（适合类别不平衡场景）
- **类别权重**：`class_weight='balanced'` 自动调整正负样本权重

#### 3.2.3 数据集划分
- 训练集：70%
- 测试集：30%
- 分层抽样：保持正负样本比例一致

### 3.3 模型评估

#### 3.3.1 评估指标
- **分类报告**：精确率(Precision)、召回率(Recall)、F1-score
- **AUC-ROC**：衡量模型区分正负样本的能力
- **混淆矩阵**：直观展示预测结果分布

#### 3.3.2 可解释性分析
- **特征重要性图**：随机森林内置的特征重要性排序
- **SHAP汇总图**：展示各特征对模型预测的全局影响
- **SHAP瀑布图**：展示单个可疑账户的预测解释

---

## 4. 数据流程

### 4.1 数据来源
- **数据集**：AMLsim 合成数据
- **路径**：`E:\Document\毕业论文——张清祥\AMLsim数据集\10Kvertices-1Medges\10Kvertices-1Medges`
- **规模**：10,000个账户，约100万笔交易

### 4.2 数据文件

#### accounts.csv（账户表）
| 字段 | 说明 |
|------|------|
| ACCOUNT_ID | 账户唯一标识 |
| INIT_BALANCE | 初始余额 |
| COUNTRY | 所属国家 |
| ACCOUNT_TYPE | 账户类型 |
| TX_BEHAVIOR_ID | 交易行为模式ID |
| IS_FRAUD | 是否为欺诈账户（标签） |

#### transactions.csv（交易表）
| 字段 | 说明 |
|------|------|
| TIMESTAMP | 交易时间戳 |
| TX_ID | 交易唯一标识 |
| SENDER_ACCOUNT_ID | 发送方账户ID |
| RECEIVER_ACCOUNT_ID | 接收方账户ID |
| TX_AMOUNT | 交易金额 |
| TX_TYPE | 交易类型 |
| IS_FRAUD | 是否为欺诈交易 |
| ALERT_ID | 关联的警报ID |

### 4.3 处理流程

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  transactions   │     │    accounts     │     │   特征工程      │
│     .csv        │────▶│     .csv        │────▶│  (18维特征)     │
│  (交易数据)      │     │  (账户标签)      │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                              ┌──────────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │   网络拓扑构建   │
                    │  (NetworkX图)   │
                    │  PageRank/聚类   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐     ┌─────────────────┐
                    │   训练集(70%)    │────▶│  随机森林模型   │
                    │   测试集(30%)    │     │  + 网格搜索     │
                    └─────────────────┘     └────────┬────────┘
                                                     │
                    ┌─────────────────┐              │
                    │  混淆矩阵/ROC    │◀─────────────┘
                    │  特征重要性      │
                    │  SHAP可解释性    │
                    └─────────────────┘
```

---

## 5. 可视化输出

### 5.1 混淆矩阵 (confusion_matrix.png)
- **用途**：展示模型在测试集上的预测准确性
- **内容**：真实标签 vs 预测标签的交叉表
- **解读**：对角线数值越高表示分类越准确

### 5.2 特征重要性 (feature_importance.png)
- **用途**：识别对模型预测贡献最大的特征
- **算法**：随机森林基于基尼不纯度减少计算
- **典型重要特征**：
  - 交易统计特征（send_sum, recv_sum）
  - 网络拓扑特征（pagerank, degree）
  - 账户初始余额（INIT_BALANCE）

### 5.3 SHAP汇总图 (shap_summary.png)
- **用途**：全局解释模型行为
- **展示**：
  - 每个特征的SHAP值分布（ beeswarm 图）
  - 特征值高低对预测的影响方向
  - 特征重要性排序

### 5.4 SHAP瀑布图 (shap_waterfall.png)
- **用途**：局部解释单个预测
- **场景**：展示一个被正确识别的可疑账户
- **内容**：
  - 基准值（base value）
  - 各特征的贡献值（推动预测向正类或负类）
  - 最终预测值

---

## 6. 使用指南

### 6.1 环境要求
```bash
pip install pandas numpy networkx scikit-learn matplotlib seaborn shap
```

### 6.2 运行步骤
1. **数据准备**：确保 AMLsim 数据集路径正确
2. **运行主程序**：
   ```bash
   python aml_model.py
   ```
3. **查看结果**：检查生成的 PNG 图片文件

### 6.3 辅助脚本使用
- **untitle1.py**：首次查看交易数据结构
- **untitle2.py**：检查数据文件完整性和基础统计
- **untitle3.py**：了解标签分布情况

---

## 7. 关键代码片段

### 7.1 特征构建核心代码
```python
# 发送方统计
sender_stats = trans.groupby('SENDER_ACCOUNT_ID').agg(
    send_sum=('TX_AMOUNT', 'sum'),
    send_mean=('TX_AMOUNT', 'mean'),
    send_max=('TX_AMOUNT', 'max'),
    send_std=('TX_AMOUNT', 'std'),
    send_count=('TX_AMOUNT', 'count')
).reset_index()

# 网络拓扑特征
G = nx.DiGraph()
for idx, row in trans.iterrows():
    G.add_edge(row['SENDER_ACCOUNT_ID'], row['RECEIVER_ACCOUNT_ID'])

pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
```

### 7.2 模型训练核心代码
```python
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}
grid = GridSearchCV(rf_base, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
```

### 7.3 SHAP解释核心代码
```python
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test_sample)
shap.summary_plot(shap_values_class, X_test_sample, feature_names=feature_cols)
```

---

## 8. 扩展建议

### 8.1 模型优化方向
- 尝试 XGBoost、LightGBM 等梯度提升模型
- 引入时序特征（交易频率、时间间隔模式）
- 使用图神经网络(GNN)替代传统网络特征

### 8.2 特征工程增强
- 添加交易金额分布的统计特征（偏度、峰度）
- 构建二阶邻居特征
- 引入社区发现算法（Louvain、Label Propagation）

### 8.3 工程化改进
- 添加日志记录
- 实现模型版本管理
- 封装为可复用的 Python 包

---

*文档生成时间：2026-05-10*
*项目作者：张清祥（毕业论文项目）*
