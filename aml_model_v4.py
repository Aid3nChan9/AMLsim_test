"""
AML Detection Model v4.0
新增功能：
1. 两种决策阈值优化策略（最大化召回率 / 最大化F2）
2. 新增特征：资金周转率、平均最短路径长度
3. 参数化设计，便于调优
使用方法：
    python aml_model_v4.py --data_dir "E:/path/to/AMLsim_data_folder" --strategy recall --recall_target 0.95
"""

import os
import argparse
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, precision_score, recall_score, fbeta_score)
from imblearn.over_sampling import BorderlineSMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_data(data_dir):
    """加载交易和账户数据"""
    trans_path = os.path.join(data_dir, 'transactions.csv')
    acc_path = os.path.join(data_dir, 'accounts.csv')
    if not os.path.exists(trans_path) or not os.path.exists(acc_path):
        raise FileNotFoundError(f"在 {data_dir} 中找不到 transactions.csv 或 accounts.csv")
    trans = pd.read_csv(trans_path)
    accounts = pd.read_csv(acc_path)
    print(f"加载交易数据：{len(trans):,} 笔，账户数据：{len(accounts):,} 个")
    return trans, accounts

def calculate_advanced_features(trans, accounts, G=None):
    """
    计算高级特征：资金周转率、平均最短路径长度
    """
    print("   计算资金周转率特征...")
    # 计算资金周转率
    inflow_sum = trans.groupby('RECEIVER_ACCOUNT_ID')['TX_AMOUNT'].sum().to_dict()
    outflow_sum = trans.groupby('SENDER_ACCOUNT_ID')['TX_AMOUNT'].sum().to_dict()
    
    turnover_ratios = {}
    for acc in accounts['ACCOUNT_ID'].unique():
        outflow = outflow_sum.get(acc, 0)
        inflow = inflow_sum.get(acc, 0)
        if inflow > 0:
            turnover_ratios[acc] = outflow / inflow  # 资金周转率：流出/流入
        else:
            turnover_ratios[acc] = 0
    
    # 计算平均最短路径长度（采样加速）
    print("   计算平均最短路径长度特征（采样节点）...")
    nodes = list(accounts['ACCOUNT_ID'].values)
    # 对于大图，只对部分节点计算（这里采样1000个节点或全部节点的20%）
    sample_size = min(2000, max(500, int(len(nodes) * 0.2)))
    sample_nodes = np.random.choice(nodes, size=sample_size, replace=False)
    
    avg_path_lengths = {}
    for node in sample_nodes:
        if G is not None and node in G:
            try:
                # 使用 limited 模式，避免在大图上计算过久
                lengths = nx.single_source_shortest_path_length(G, node, cutoff=5)
                if len(lengths) > 1:
                    avg_length = sum(lengths.values()) / (len(lengths) - 1)
                else:
                    avg_length = 0
            except:
                avg_length = 0
        else:
            avg_length = 0
        avg_path_lengths[node] = avg_length
    
    # 为所有节点填充（未采样的节点使用整体中位数）
    all_avg_lengths = {}
    median_length = np.median(list(avg_path_lengths.values())) if avg_path_lengths else 0
    for node in nodes:
        all_avg_lengths[node] = avg_path_lengths.get(node, median_length)
    
    return turnover_ratios, all_avg_lengths

def extract_features(trans, accounts):
    """特征工程：统计特征 + 网络拓扑特征 + 高级特征 + 类别编码"""
    print("  构建交易统计特征...")
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
    print("  构建交易网络有向图...")
    G = nx.DiGraph()
    for _, row in trans.iterrows():
        G.add_edge(row['SENDER_ACCOUNT_ID'], row['RECEIVER_ACCOUNT_ID'])
    
    nodes = list(accounts['ACCOUNT_ID'].values)
    print("  计算网络特征（出入度、PageRank、聚集系数）...")
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
    
    # 高级特征（新增）
    turnover_ratios, avg_path_lengths = calculate_advanced_features(trans, accounts, G)
    account_features['turnover_ratio'] = account_features['ACCOUNT_ID'].map(turnover_ratios).fillna(0)
    account_features['avg_path_length'] = account_features['ACCOUNT_ID'].map(avg_path_lengths).fillna(0)
    
    # 类别编码
    print("  类别特征编码...")
    le_country = LabelEncoder()
    le_type = LabelEncoder()
    le_behavior = LabelEncoder()
    account_features['country_enc'] = le_country.fit_transform(account_features['COUNTRY'].astype(str))
    account_features['type_enc'] = le_type.fit_transform(account_features['ACCOUNT_TYPE'].astype(str))
    account_features['behavior_enc'] = le_behavior.fit_transform(account_features['TX_BEHAVIOR_ID'].fillna(-1).astype(int))
    
    # 特征列表（新增 turnover_ratio 和 avg_path_length）
    feature_cols = [
        'INIT_BALANCE', 'send_sum', 'send_mean', 'send_max', 'send_std', 'send_count',
        'recv_sum', 'recv_mean', 'recv_max', 'recv_std', 'recv_count',
        'out_in_ratio', 'net_flow',
        'out_degree', 'in_degree', 'total_degree', 'pagerank', 'clustering',
        'turnover_ratio', 'avg_path_length',
        'country_enc', 'type_enc', 'behavior_enc'
    ]
    X = account_features[feature_cols].fillna(0)
    y = account_features['IS_FRAUD']
    return X, y, feature_cols, (le_country, le_type, le_behavior)

def find_optimal_threshold(y_val, y_val_proba, strategy='recall', recall_target=0.95, beta=2.0):
    """
    根据策略找到最优决策阈值
    strategy: 'recall' - 在达到目标召回率的前提下最大化精确率
              'fbeta'  - 最大化 F-beta 分数（beta越大召回率权重越高）
    """
    thresholds = np.linspace(0.1, 0.95, 50)
    
    if strategy == 'recall':
        best_th = 0.5
        best_precision = 0
        for th in thresholds:
            y_pred = (y_val_proba >= th).astype(int)
            recall = recall_score(y_val, y_pred)
            if recall >= recall_target:
                precision = precision_score(y_val, y_pred)
                if precision > best_precision:
                    best_precision = precision
                    best_th = th
        print(f"阈值策略: 召回率目标={recall_target:.0%} → 最佳阈值={best_th:.2f} (召回率={recall_score(y_val, (y_val_proba>=best_th).astype(int)):.4f}, 精确率={best_precision:.4f})")
        return best_th
    
    elif strategy == 'fbeta':
        best_th = 0.5
        best_fbeta = 0
        for th in thresholds:
            y_pred = (y_val_proba >= th).astype(int)
            fbeta = fbeta_score(y_val, y_pred, beta=beta)
            if fbeta > best_fbeta:
                best_fbeta = fbeta
                best_th = th
        print(f"阈值策略: F{beta}最大化 → 最佳阈值={best_th:.2f} (F{beta}={best_fbeta:.4f})")
        return best_th
    
    else:
        print(f"未知策略 {strategy}，使用默认阈值 0.5")
        return 0.5

def run(data_dir, strategy='recall', recall_target=0.95, beta=2.0):
    """主流程"""
    print(f"\n{'='*60}")
    print(f"AML Detection Model v4.0")
    print(f"数据集: {data_dir}")
    print(f"阈值策略: {strategy}" + (f" (召回率目标={recall_target:.0%})" if strategy=='recall' else f" (beta={beta})"))
    print(f"{'='*60}\n")
    
    # 加载数据
    trans, accounts = load_data(data_dir)
    
    # 特征提取
    X, y, feature_cols, encoders = extract_features(trans, accounts)
    print(f"\n特征矩阵形状: {X.shape}, 正样本比例: {y.mean():.4f}")
    
    # 数据集划分 (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    print(f"训练集: {len(X_train)} (可疑: {y_train.sum()})")
    print(f"验证集: {len(X_val)} (可疑: {y_val.sum()})")
    print(f"测试集: {len(X_test)} (可疑: {y_test.sum()})")
    
    # Borderline-SMOTE 过采样
    print("\n应用 Borderline-SMOTE 过采样...")
    smote = BorderlineSMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"过采样后训练集: {len(X_train_res)} (可疑: {y_train_res.sum()})")
    
    # 随机森林 + 网格搜索
    print("\n训练随机森林模型 (GridSearchCV)...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced_subsample')
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train_res, y_train_res)
    
    best_rf = grid.best_estimator_
    print(f"最佳参数: {grid.best_params_}")
    
    # 验证集预测概率
    y_val_proba = best_rf.predict_proba(X_val)[:, 1]
    
    # 寻找最优阈值
    best_th = find_optimal_threshold(y_val, y_val_proba, strategy, recall_target, beta)
    
    # 测试集最终评估
    y_test_proba = best_rf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= best_th).astype(int)
    
    print("\n" + "="*60)
    print("最终测试集评估结果")
    print("="*60)
    print(classification_report(y_test, y_test_pred, target_names=['正常', '可疑']))
    print(f"AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
    print(f"召回率: {recall_score(y_test, y_test_pred):.4f}")
    print(f"精确率: {precision_score(y_test, y_test_pred):.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['正常','可疑'], yticklabels=['正常','可疑'])
    plt.xlabel('预测'); plt.ylabel('真实'); plt.title(f'测试集混淆矩阵 (阈值={best_th:.2f})')
    plt.savefig('confusion_matrix_v4.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 特征重要性
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12,8))
    plt.title('随机森林特征重要性 (包含新增的高级特征)')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_v4.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n特征重要性图已保存为 feature_importance_v4.png")
    
    # SHAP 分析（可选）
    print("\n计算 SHAP 值...")
    X_sample = X_test.sample(min(200, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(best_rf)
    shap_values = explainer.shap_values(X_sample)
    shap_values_class = shap_values[1] if isinstance(shap_values, list) else shap_values[:,:,1]
    shap.summary_plot(shap_values_class, X_sample, feature_names=feature_cols, show=False)
    plt.savefig('shap_summary_v4.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图表已生成: confusion_matrix_v4.png, feature_importance_v4.png, shap_summary_v4.png")
    
    return best_rf, best_th

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Detection Model v4.0")
    parser.add_argument('--data_dir', type=str, required=True, help='数据文件夹路径')
    parser.add_argument('--strategy', type=str, default='recall', choices=['recall', 'fbeta'], help='阈值优化策略')
    parser.add_argument('--recall_target', type=float, default=0.95, help='召回率目标 (0-1)')
    parser.add_argument('--beta', type=float, default=2.0, help='F-beta 分数中的 beta 参数')
    args = parser.parse_args()
    run(args.data_dir, args.strategy, args.recall_target, args.beta)