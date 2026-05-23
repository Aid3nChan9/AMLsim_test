"""
AML Detection Model v3.0
支持手动指定数据集路径，自动完成特征工程、模型训练、评估和可解释性分析。
使用方法：
    python aml_model_v3.py --data_dir "E:/path/to/your/AMLsim_data_folder"
或直接在代码中修改 default_data_dir 变量。
"""

import os
import argparse
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import BorderlineSMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_data(data_dir):
    """从指定目录加载 transactions.csv 和 accounts.csv"""
    trans_path = os.path.join(data_dir, 'transactions.csv')
    acc_path = os.path.join(data_dir, 'accounts.csv')
    if not os.path.exists(trans_path) or not os.path.exists(acc_path):
        raise FileNotFoundError(f"在 {data_dir} 中找不到 transactions.csv 或 accounts.csv")
    trans = pd.read_csv(trans_path)
    accounts = pd.read_csv(acc_path)
    print(f"加载交易数据：{len(trans)} 笔，账户数据：{len(accounts)} 个")
    return trans, accounts

def extract_features(trans, accounts):
    """特征工程：统计特征 + 网络拓扑特征 + 类别编码"""
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
    
    # 合并
    account_features = accounts[['ACCOUNT_ID', 'IS_FRAUD', 'INIT_BALANCE', 'COUNTRY', 'ACCOUNT_TYPE', 'TX_BEHAVIOR_ID']].copy()
    account_features = account_features.merge(sender_stats, on='ACCOUNT_ID', how='left')
    account_features = account_features.merge(receiver_stats, on='ACCOUNT_ID', how='left')
    
    fill_cols = ['send_sum', 'send_mean', 'send_max', 'send_std', 'send_count',
                 'recv_sum', 'recv_mean', 'recv_max', 'recv_std', 'recv_count']
    account_features[fill_cols] = account_features[fill_cols].fillna(0)
    
    # 衍生特征
    account_features['out_in_ratio'] = account_features['send_count'] / (account_features['recv_count'] + 1e-6)
    account_features['net_flow'] = account_features['send_sum'] - account_features['recv_sum']
    
    # 网络拓扑特征
    print("构建有向图...")
    G = nx.DiGraph()
    for _, row in trans.iterrows():
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
    X = account_features[feature_cols].fillna(0)
    y = account_features['IS_FRAUD']
    return X, y, feature_cols, (le_country, le_type, le_behavior)

def run(data_dir):
    """主流程：加载数据、特征提取、训练、评估、生成图表"""
    print(f"=== 开始处理数据集: {data_dir} ===")
    trans, accounts = load_data(data_dir)
    X, y, feature_cols, encoders = extract_features(trans, accounts)
    print(f"特征矩阵形状: {X.shape}, 正样本比例: {y.mean():.4f}")
    
    # 划分训练/验证/测试集 (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    # Borderline-SMOTE 过采样
    smote = BorderlineSMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"过采样后训练集大小: {len(X_train_res)}, 正样本: {y_train_res.sum()}")
    
    # 随机森林 + 网格搜索
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced_subsample')
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train_res, y_train_res)
    best_rf = grid.best_estimator_
    print(f"最佳参数: {grid.best_params_}")
    
    # 验证集调优阈值
    y_val_proba = best_rf.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 9)
    best_th = 0.5
    best_f1 = 0
    for th in thresholds:
        y_pred = (y_val_proba >= th).astype(int)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    print(f"最佳阈值: {best_th:.2f} (验证集F1={best_f1:.4f})")
    
    # 测试集评估
    y_test_proba = best_rf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= best_th).astype(int)
    print("\n=== 测试集分类报告 ===")
    print(classification_report(y_test, y_test_pred, target_names=['正常', '可疑']))
    print(f"AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['正常','可疑'], yticklabels=['正常','可疑'])
    plt.xlabel('预测'); plt.ylabel('真实'); plt.title('测试集混淆矩阵')
    plt.savefig('confusion_matrix_v3.png', dpi=150, bbox_inches='tight')
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
    plt.savefig('feature_importance_v3.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("特征重要性图已保存为 feature_importance_v3.png")
    
    # SHAP（可选，耗时）
    print("计算SHAP值（子集200样本）...")
    X_sample = X_test.sample(min(200, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(best_rf)
    shap_values = explainer.shap_values(X_sample)
    shap_values_class = shap_values[1] if isinstance(shap_values, list) else shap_values[:,:,1]
    shap.summary_plot(shap_values_class, X_sample, feature_names=feature_cols, show=False)
    plt.savefig('shap_summary_v3.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 瀑布图
    y_pred_sample = best_rf.predict(X_sample)
    y_true_sample = y_test.loc[X_sample.index]
    pos_correct = (y_true_sample == 1) & (y_pred_sample == 1)
    if pos_correct.any():
        idx = X_sample.index[pos_correct][0]
        row_loc = X_sample.index.get_loc(idx)
        exp_val = explainer.expected_value
        base_val = float(exp_val[1]) if isinstance(exp_val, (list, np.ndarray)) and len(exp_val)>=2 else float(exp_val)
        exp = shap.Explanation(values=shap_values_class[row_loc],
                               base_values=base_val,
                               data=X_sample.iloc[row_loc, :].values,
                               feature_names=feature_cols)
        shap.waterfall_plot(exp, show=False)
        plt.savefig('shap_waterfall_v3.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("SHAP瀑布图已保存")
    else:
        print("未找到正确预测的可疑样本，跳过瀑布图")
    
    print("全部完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Detection Model")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to folder containing transactions.csv and accounts.csv')
    args = parser.parse_args()
    run(args.data_dir)