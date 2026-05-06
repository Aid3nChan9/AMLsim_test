import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import shap

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# ===================== 1. 加载数据 =====================
data_dir = r"E:\Document\毕业论文——张清祥\AMLsim数据集\10Kvertices-1Medges\10Kvertices-1Medges"

print("加载交易数据...")
trans = pd.read_csv(data_dir + "/transactions.csv")
print(f"交易数据量: {len(trans)} 笔")

print("加载账户数据...")
accounts = pd.read_csv(data_dir + "/accounts.csv")
print(f"账户数量: {len(accounts)}")
print(f"可疑账户占比: {accounts['IS_FRAUD'].mean():.2%}")

# ===================== 2. 交易行为统计特征（按账户聚合）=====================
print("\n构建账户级交易统计特征...")

# 发送方统计
sender_stats = trans.groupby('SENDER_ACCOUNT_ID').agg(
    send_sum=('TX_AMOUNT', 'sum'),
    send_mean=('TX_AMOUNT', 'mean'),
    send_max=('TX_AMOUNT', 'max'),
    send_std=('TX_AMOUNT', 'std'),
    send_count=('TX_AMOUNT', 'count')
).reset_index().rename(columns={'SENDER_ACCOUNT_ID': 'ACCOUNT_ID'})

# 接收方统计
receiver_stats = trans.groupby('RECEIVER_ACCOUNT_ID').agg(
    recv_sum=('TX_AMOUNT', 'sum'),
    recv_mean=('TX_AMOUNT', 'mean'),
    recv_max=('TX_AMOUNT', 'max'),
    recv_std=('TX_AMOUNT', 'std'),
    recv_count=('TX_AMOUNT', 'count')
).reset_index().rename(columns={'RECEIVER_ACCOUNT_ID': 'ACCOUNT_ID'})

# 合并发送和接收统计
account_features = accounts[['ACCOUNT_ID', 'IS_FRAUD', 'INIT_BALANCE', 'COUNTRY', 'ACCOUNT_TYPE', 'TX_BEHAVIOR_ID']].copy()
account_features = account_features.merge(sender_stats, on='ACCOUNT_ID', how='left')
account_features = account_features.merge(receiver_stats, on='ACCOUNT_ID', how='left')

# 填充缺失值（某些账户可能从未发送或从未接收）
fill_cols = ['send_sum', 'send_mean', 'send_max', 'send_std', 'send_count',
             'recv_sum', 'recv_mean', 'recv_max', 'recv_std', 'recv_count']
account_features[fill_cols] = account_features[fill_cols].fillna(0)

# 衍生特征：出入度比（发收比）
account_features['out_in_ratio'] = account_features['send_count'] / (account_features['recv_count'] + 1e-6)
# 净流量（发送总额 - 接收总额）
account_features['net_flow'] = account_features['send_sum'] - account_features['recv_sum']

# ===================== 3. 网络拓扑特征（基于NetworkX）=====================
print("构建交易网络拓扑特征（可能需要几分钟）...")

# 构建有向图（只使用部分交易？为了速度可以限制交易数量，但这里全量132万笔可能稍慢）
# 如果你的电脑内存足够（16G），全量构建应该没问题。如果太慢，可以采样一部分交易。
G = nx.DiGraph()
# 添加边（发送方 -> 接收方，可以多重边聚合为单边权重，但简单加边即可）
for idx, row in trans.iterrows():
    G.add_edge(row['SENDER_ACCOUNT_ID'], row['RECEIVER_ACCOUNT_ID'])

# 计算每个节点的特征（只对账户表中存在的账户）
nodes = list(accounts['ACCOUNT_ID'].values)
# 出度、入度、总度
out_degree = dict(G.out_degree(nodes))
in_degree = dict(G.in_degree(nodes))
# PageRank
pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
# 聚集系数（针对有向图，使用局部聚类）
clustering = nx.clustering(G.to_undirected())  # 转为无向图计算聚集系数（更快）

# 将结果存入DataFrame
network_feat = pd.DataFrame({'ACCOUNT_ID': nodes})
network_feat['out_degree'] = network_feat['ACCOUNT_ID'].map(out_degree).fillna(0)
network_feat['in_degree'] = network_feat['ACCOUNT_ID'].map(in_degree).fillna(0)
network_feat['total_degree'] = network_feat['out_degree'] + network_feat['in_degree']
network_feat['pagerank'] = network_feat['ACCOUNT_ID'].map(pagerank).fillna(0)
network_feat['clustering'] = network_feat['ACCOUNT_ID'].map(clustering).fillna(0)

# 合并网络特征
account_features = account_features.merge(network_feat, on='ACCOUNT_ID', how='left')

# ===================== 4. 类别特征编码 =====================
le_country = LabelEncoder()
le_type = LabelEncoder()
le_behavior = LabelEncoder()

account_features['country_enc'] = le_country.fit_transform(account_features['COUNTRY'].astype(str))
account_features['type_enc'] = le_type.fit_transform(account_features['ACCOUNT_TYPE'].astype(str))
account_features['behavior_enc'] = le_behavior.fit_transform(account_features['TX_BEHAVIOR_ID'].fillna(-1).astype(int))

# ===================== 5. 选择特征和标签 =====================
feature_cols = [
    'INIT_BALANCE', 'send_sum', 'send_mean', 'send_max', 'send_std', 'send_count',
    'recv_sum', 'recv_mean', 'recv_max', 'recv_std', 'recv_count',
    'out_in_ratio', 'net_flow',
    'out_degree', 'in_degree', 'total_degree', 'pagerank', 'clustering',
    'country_enc', 'type_enc', 'behavior_enc'
]
X = account_features[feature_cols]
y = account_features['IS_FRAUD']

# 检查缺失值
print("缺失值检查：", X.isnull().sum().sum())
if X.isnull().sum().sum() > 0:
    X = X.fillna(0)

print(f"特征矩阵形状: {X.shape}")
print(f"正样本比例: {y.mean():.4f}")

# ===================== 6. 划分训练集和测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ===================== 7. 随机森林模型（带简单网格搜索）=====================
print("\n训练随机森林模型...")
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}
grid = GridSearchCV(rf_base, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
print(f"最佳参数: {grid.best_params_}")

# ===================== 8. 模型评估 =====================
# ===================== 8. 模型评估 =====================
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

print("\n=== 分类报告 ===")
print(classification_report(y_test, y_pred, target_names=['正常', '可疑']))
print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['正常','可疑'], yticklabels=['正常','可疑'])
plt.xlabel('预测')
plt.ylabel('真实')
plt.title('混淆矩阵')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()   # 关闭当前图，避免干扰后续绘图

# ===================== 9. 特征重要性 =====================
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
plt.title('随机森林特征重要性')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("特征重要性图已保存为 feature_importance.png")

# ===================== 10. SHAP 可解释性（稳健版）======================
print("\n计算SHAP值（可能需要几分钟）...")
# 为了加速，使用测试集的一个子集（例如200个样本）
X_test_sample = X_test.sample(min(200, len(X_test)), random_state=42)

explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test_sample)

# 处理二分类情况：shap_values 可能是列表 [shap_0, shap_1] 或 三维数组
if isinstance(shap_values, list):
    # 正类的 shap 值
    shap_values_class = shap_values[1]
elif len(shap_values.shape) == 3:
    # 形状 (n_samples, n_features, n_classes)
    shap_values_class = shap_values[:, :, 1]
else:
    shap_values_class = shap_values
    print("警告：shap_values 的维度为", shap_values.shape)

print(f"SHAP values 形状: {shap_values_class.shape}, 特征矩阵形状: {X_test_sample.shape}")
assert shap_values_class.shape[1] == X_test_sample.shape[1], "特征数量不匹配"

# SHAP 汇总图（beeswarm plot）
shap.summary_plot(shap_values_class, X_test_sample, feature_names=feature_cols, show=False)
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("SHAP 汇总图已保存为 shap_summary.png")

# 挑选一个预测正确的可疑账户绘制瀑布图
y_pred_sample = best_rf.predict(X_test_sample)
y_true_sample = y_test.loc[X_test_sample.index]

# 找到真实为可疑且预测也为可疑的样本
pos_correct = (y_true_sample == 1) & (y_pred_sample == 1)
if pos_correct.any():
    idx = X_test_sample.index[pos_correct][0]         # 第一个满足条件的账户ID
    row_idx = X_test_sample.index.get_loc(idx)        # 在 X_test_sample 中的行位置
    
    # 正确处理 base_values（兼容列表或数组）
    exp_val = explainer.expected_value
    if isinstance(exp_val, (list, np.ndarray)):
        if len(exp_val) >= 2:
            base_val = float(exp_val[1])   # 正类（可疑）的基准值
        else:
            base_val = float(exp_val[0])
    else:
        base_val = float(exp_val)
    
    # 创建 Explanation 对象并绘图
    exp = shap.Explanation(values=shap_values_class[row_idx],
                           base_values=base_val,
                           data=X_test_sample.iloc[row_idx, :].values,
                           feature_names=feature_cols)
    shap.waterfall_plot(exp, show=False)
    plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP 瀑布图已保存为 shap_waterfall.png")
else:
    print("未找到预测正确的可疑样本，跳过瀑布图")