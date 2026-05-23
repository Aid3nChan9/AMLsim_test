import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import BorderlineSMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# 设置中文字体（避免乱码）
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# ===================== 1. 加载数据 =====================
data_dir = r"E:\Document\毕业论文——张清祥\AMLsim数据集\10Kvertices-1Medges\10Kvertices-1Medges"

print("加载交易数据...")
trans = pd.read_csv(data_dir + "/transactions.csv")
print(f"交易数据量: {len(trans)} 笔")

print("加载账户数据...")
accounts = pd.read_csv(data_dir + "/accounts.csv")
print(f"账户数量: {len(accounts)}")
print(f"可疑账户占比: {accounts['IS_FRAUD'].mean():.2%}")

# ===================== 2. 特征工程（与之前相同）=====================
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

account_features = accounts[['ACCOUNT_ID', 'IS_FRAUD', 'INIT_BALANCE', 'COUNTRY', 'ACCOUNT_TYPE', 'TX_BEHAVIOR_ID']].copy()
account_features = account_features.merge(sender_stats, on='ACCOUNT_ID', how='left')
account_features = account_features.merge(receiver_stats, on='ACCOUNT_ID', how='left')

fill_cols = ['send_sum', 'send_mean', 'send_max', 'send_std', 'send_count',
             'recv_sum', 'recv_mean', 'recv_max', 'recv_std', 'recv_count']
account_features[fill_cols] = account_features[fill_cols].fillna(0)

account_features['out_in_ratio'] = account_features['send_count'] / (account_features['recv_count'] + 1e-6)
account_features['net_flow'] = account_features['send_sum'] - account_features['recv_sum']

print("构建网络拓扑特征...")
G = nx.DiGraph()
for idx, row in trans.iterrows():
    G.add_edge(row['SENDER_ACCOUNT_ID'], row['RECEIVER_ACCOUNT_ID'])

nodes = list(accounts['ACCOUNT_ID'].values)
out_degree = dict(G.out_degree(nodes))
in_degree = dict(G.in_degree(nodes))
pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
clustering = nx.clustering(G.to_undirected())

network_feat = pd.DataFrame({'ACCOUNT_ID': nodes})
network_feat['out_degree'] = network_feat['ACCOUNT_ID'].map(out_degree).fillna(0)
network_feat['in_degree'] = network_feat['ACCOUNT_ID'].map(in_degree).fillna(0)
network_feat['total_degree'] = network_feat['out_degree'] + network_feat['in_degree']
network_feat['pagerank'] = network_feat['ACCOUNT_ID'].map(pagerank).fillna(0)
network_feat['clustering'] = network_feat['ACCOUNT_ID'].map(clustering).fillna(0)

account_features = account_features.merge(network_feat, on='ACCOUNT_ID', how='left')

# 类别编码
le_country = LabelEncoder()
le_type = LabelEncoder()
le_behavior = LabelEncoder()
account_features['country_enc'] = le_country.fit_transform(account_features['COUNTRY'].astype(str))
account_features['type_enc'] = le_type.fit_transform(account_features['ACCOUNT_TYPE'].astype(str))
account_features['behavior_enc'] = le_behavior.fit_transform(account_features['TX_BEHAVIOR_ID'].fillna(-1).astype(int))

feature_cols = [
    'INIT_BALANCE', 'send_sum', 'send_mean', 'send_max', 'send_std', 'send_count',
    'recv_sum', 'recv_mean', 'recv_max', 'recv_std', 'recv_count',
    'out_in_ratio', 'net_flow',
    'out_degree', 'in_degree', 'total_degree', 'pagerank', 'clustering',
    'country_enc', 'type_enc', 'behavior_enc'
]
X = account_features[feature_cols]
y = account_features['IS_FRAUD']
X = X.fillna(0)

print(f"特征矩阵形状: {X.shape}, 正样本比例: {y.mean():.4f}")

# ===================== 3. 划分训练、验证、测试集 =====================
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)  # 0.25 * 0.8 = 0.2

print(f"训练集大小: {len(X_train)} (可疑: {y_train.sum()})")
print(f"验证集大小: {len(X_val)} (可疑: {y_val.sum()})")
print(f"测试集大小: {len(X_test)} (可疑: {y_test.sum()})")

# ===================== 4. Borderline-SMOTE 过采样 =====================
print("\n应用 Borderline-SMOTE 过采样...")
smote = BorderlineSMOTE(random_state=42, kind='borderline-1')
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"过采样后训练集大小: {len(X_train_res)}, 正样本数: {y_train_res.sum()}")

# ===================== 5. 随机森林 + 网格搜索 =====================
print("\n训练随机森林模型 (GridSearchCV)...")
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced_subsample')
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}
grid = GridSearchCV(rf_base, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid.fit(X_train_res, y_train_res)

best_rf = grid.best_estimator_
print(f"最佳参数: {grid.best_params_}")

# ===================== 6. 验证集上调整阈值（可选） =====================
y_val_proba = best_rf.predict_proba(X_val)[:, 1]
best_threshold = 0.5
# 简单搜索最佳阈值（根据验证集 F1 或 recall）
thresholds = np.linspace(0.1, 0.9, 9)
best_f1 = 0
for th in thresholds:
    y_pred_th = (y_val_proba >= th).astype(int)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_val, y_pred_th)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = th
print(f"验证集最佳阈值: {best_threshold:.2f} (F1={best_f1:.4f})")

# ===================== 7. 测试集最终评估 =====================
y_test_proba = best_rf.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= best_threshold).astype(int)

print("\n=== 测试集分类报告 (使用最佳阈值) ===")
print(classification_report(y_test, y_test_pred, target_names=['正常', '可疑']))
print(f"AUC: {roc_auc_score(y_test, y_test_proba):.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['正常','可疑'], yticklabels=['正常','可疑'])
plt.xlabel('预测')
plt.ylabel('真实')
plt.title('测试集混淆矩阵')
plt.savefig('confusion_matrix_v2.png', dpi=150, bbox_inches='tight')
plt.close()

# 特征重要性
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
plt.title('随机森林特征重要性')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("特征重要性图已保存为 feature_importance_v2.png")

# ===================== 8. SHAP 可解释性 (可选，较慢) =====================
print("\n计算 SHAP 值 (使用测试集子集 200 样本)...")
X_test_sample = X_test.sample(min(200, len(X_test)), random_state=42)
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test_sample)
if isinstance(shap_values, list):
    shap_values_class = shap_values[1]
else:
    shap_values_class = shap_values[:, :, 1]
print(f"SHAP values 形状: {shap_values_class.shape}")

shap.summary_plot(shap_values_class, X_test_sample, feature_names=feature_cols, show=False)
plt.savefig('shap_summary_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("SHAP 汇总图已保存为 shap_summary_v2.png")

# 瀑布图
y_pred_sample = best_rf.predict(X_test_sample)
y_true_sample = y_test.loc[X_test_sample.index]
pos_correct = (y_true_sample == 1) & (y_pred_sample == 1)
if pos_correct.any():
    idx = X_test_sample.index[pos_correct][0]
    row_idx = X_test_sample.index.get_loc(idx)
    exp_val = explainer.expected_value
    if isinstance(exp_val, (list, np.ndarray)):
        base_val = float(exp_val[1]) if len(exp_val) >= 2 else float(exp_val[0])
    else:
        base_val = float(exp_val)
    exp = shap.Explanation(values=shap_values_class[row_idx],
                           base_values=base_val,
                           data=X_test_sample.iloc[row_idx, :].values,
                           feature_names=feature_cols)
    shap.waterfall_plot(exp, show=False)
    plt.savefig('shap_waterfall_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP 瀑布图已保存为 shap_waterfall_v2.png")
else:
    print("未找到正确的可疑样本，跳过瀑布图")

# ===================== 9. (可选) 外部新数据集评估函数 =====================
def evaluate_on_new_data(new_data_dir, trained_model, feature_cols, le_country, le_type, le_behavior):
    """
    加载新的 AMLSim 生成的数据集，并进行相同的特征工程，最后评估模型性能。
    """
    print(f"\n评估外部数据集: {new_data_dir}")
    trans_new = pd.read_csv(f"{new_data_dir}/transactions.csv")
    accounts_new = pd.read_csv(f"{new_data_dir}/accounts.csv")
    
    # 重复上述特征工程步骤 (为简洁，这里只列出必要部分，实际需复制相同代码)
    # ... 建议你将特征工程封装成一个函数，此处调用。
    # 因为代码较长，这里给出框架，你需要根据实际情况补充。
    # 注意：类别编码要使用训练时拟合好的 LabelEncoder (le_country, le_type, le_behavior)
    # 对于新出现的类别，可以映射为 -1。
    
    # 简单示例：假设你已经实现了 get_features() 函数
    # X_new = get_features(trans_new, accounts_new, le_country, le_type, le_behavior)
    # y_new = accounts_new['IS_FRAUD']
    # X_new = X_new[feature_cols].fillna(0)
    # y_pred = trained_model.predict(X_new)
    # print(classification_report(y_new, y_pred))
    pass

# 如果生成了新数据，可以调用：
# evaluate_on_new_data(r"E:\path\to\new_AMLsim_output", best_rf, feature_cols, le_country, le_type, le_behavior)

print("\n全部完成！最终模型已评估，图表已生成。")