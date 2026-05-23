"""
AML Detection Model v10.0
- 混合采样 SMOTETomek（清理边界样本）
- 召回率优先阈值优化（默认 80%）
- 集成学习：随机森林 + XGBoost（可配置权重）
- 静默账户后处理
- 无高级特征（避免特征空间不一致）
使用方法：
    python aml_model_v10.py --data_dir <路径> [--recall_target 0.80] [--rf_weight 0.5] [--xgb_weight 0.5]
"""

import os
import argparse
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, precision_score, recall_score)
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

# 可选 GPU 加速（随机森林）
try:
    from cuml.ensemble import RandomForestClassifier as CumlRandomForest
    USE_GPU_RF = True
    print("检测到 cuML，随机森林将使用 GPU")
except ImportError:
    from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
    USE_GPU_RF = False
    print("未检测到 cuML，随机森林使用 CPU")

# XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("XGBoost 已安装，将参与集成")
except ImportError:
    XGB_AVAILABLE = False
    print("未安装 XGBoost，将只使用随机森林。请运行 pip install xgboost")

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# ===================== 1. 数据加载与时间划分 =====================
def load_data(data_dir):
    trans_path = os.path.join(data_dir, 'transactions.csv')
    acc_path = os.path.join(data_dir, 'accounts.csv')
    if not os.path.exists(trans_path) or not os.path.exists(acc_path):
        raise FileNotFoundError(f"在 {data_dir} 中找不到 transactions.csv 或 accounts.csv")
    trans = pd.read_csv(trans_path)
    accounts = pd.read_csv(acc_path)
    print(f"加载交易数据：{len(trans):,} 笔，账户数据：{len(accounts):,} 个")
    return trans, accounts

def split_by_time(trans, train_ratio=0.8, val_ratio=0.1):
    trans_sorted = trans.sort_values('TIMESTAMP')
    n = len(trans_sorted)
    train_val_end = int(n * train_ratio)
    trans_train_val = trans_sorted.iloc[:train_val_end]
    trans_test = trans_sorted.iloc[train_val_end:]
    n_train_val = len(trans_train_val)
    train_end = int(n_train_val * (1 - val_ratio))
    trans_train = trans_train_val.iloc[:train_end]
    trans_val = trans_train_val.iloc[train_end:]
    print(f"按时间划分：训练集交易 {len(trans_train):,} 笔，验证集 {len(trans_val):,} 笔，测试集 {len(trans_test):,} 笔")
    return trans_train, trans_val, trans_test

# ===================== 2. 特征工程（不含高级特征）=====================
def extract_temporal_features(trans, accounts):
    trans = trans.copy()
    trans['day'] = trans['TIMESTAMP'] // 86400
    trans['is_night'] = (trans['TIMESTAMP'] % 86400) < 21600
    
    sender_group = trans.groupby('SENDER_ACCOUNT_ID')
    active_days_send = sender_group['day'].nunique().to_dict()
    daily_tx_send = (sender_group.size() / sender_group['day'].nunique()).to_dict()
    night_amount_send = trans[trans['is_night']].groupby('SENDER_ACCOUNT_ID')['TX_AMOUNT'].sum().to_dict()
    total_amount_send = sender_group['TX_AMOUNT'].sum().to_dict()
    night_ratio_send = {}
    for acc in set(list(night_amount_send.keys()) + list(total_amount_send.keys())):
        night = night_amount_send.get(acc, 0.0)
        total = total_amount_send.get(acc, 0.0)
        night_ratio_send[acc] = night / (total + 1e-8)
    daily_amount_max_send = sender_group.apply(lambda x: x.groupby('day')['TX_AMOUNT'].sum().max()).to_dict()
    daily_count_max_send = sender_group.apply(lambda x: x.groupby('day').size().max()).to_dict()
    
    receiver_group = trans.groupby('RECEIVER_ACCOUNT_ID')
    active_days_recv = receiver_group['day'].nunique().to_dict()
    daily_tx_recv = (receiver_group.size() / receiver_group['day'].nunique()).to_dict()
    night_amount_recv = trans[trans['is_night']].groupby('RECEIVER_ACCOUNT_ID')['TX_AMOUNT'].sum().to_dict()
    total_amount_recv = receiver_group['TX_AMOUNT'].sum().to_dict()
    night_ratio_recv = {}
    for acc in set(list(night_amount_recv.keys()) + list(total_amount_recv.keys())):
        night = night_amount_recv.get(acc, 0.0)
        total = total_amount_recv.get(acc, 0.0)
        night_ratio_recv[acc] = night / (total + 1e-8)
    daily_amount_max_recv = receiver_group.apply(lambda x: x.groupby('day')['TX_AMOUNT'].sum().max()).to_dict()
    daily_count_max_recv = receiver_group.apply(lambda x: x.groupby('day').size().max()).to_dict()
    
    all_accounts = accounts['ACCOUNT_ID'].unique()
    temporal_feat = pd.DataFrame({'ACCOUNT_ID': all_accounts})
    temporal_feat['active_days_send'] = temporal_feat['ACCOUNT_ID'].map(active_days_send).fillna(0)
    temporal_feat['daily_tx_send'] = temporal_feat['ACCOUNT_ID'].map(daily_tx_send).fillna(0)
    temporal_feat['night_ratio_send'] = temporal_feat['ACCOUNT_ID'].map(night_ratio_send).fillna(0)
    temporal_feat['max_daily_amount_send'] = temporal_feat['ACCOUNT_ID'].map(daily_amount_max_send).fillna(0)
    temporal_feat['max_daily_count_send'] = temporal_feat['ACCOUNT_ID'].map(daily_count_max_send).fillna(0)
    temporal_feat['active_days_recv'] = temporal_feat['ACCOUNT_ID'].map(active_days_recv).fillna(0)
    temporal_feat['daily_tx_recv'] = temporal_feat['ACCOUNT_ID'].map(daily_tx_recv).fillna(0)
    temporal_feat['night_ratio_recv'] = temporal_feat['ACCOUNT_ID'].map(night_ratio_recv).fillna(0)
    temporal_feat['max_daily_amount_recv'] = temporal_feat['ACCOUNT_ID'].map(daily_amount_max_recv).fillna(0)
    temporal_feat['max_daily_count_recv'] = temporal_feat['ACCOUNT_ID'].map(daily_count_max_recv).fillna(0)
    return temporal_feat

def build_graph_and_extract_network_features(trans_train_val, accounts):
    print("构建有向图（仅训练+验证交易）...")
    G = nx.from_pandas_edgelist(trans_train_val, 
                                source='SENDER_ACCOUNT_ID', 
                                target='RECEIVER_ACCOUNT_ID',
                                create_using=nx.DiGraph())
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
    return network_feat

def extract_features(trans_train_val, accounts):
    # 统计特征
    sender_stats = trans_train_val.groupby('SENDER_ACCOUNT_ID').agg(
        send_sum=('TX_AMOUNT', 'sum'),
        send_mean=('TX_AMOUNT', 'mean'),
        send_max=('TX_AMOUNT', 'max'),
        send_std=('TX_AMOUNT', 'std'),
        send_count=('TX_AMOUNT', 'count')
    ).reset_index().rename(columns={'SENDER_ACCOUNT_ID': 'ACCOUNT_ID'})
    receiver_stats = trans_train_val.groupby('RECEIVER_ACCOUNT_ID').agg(
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
    
    # 网络特征
    network_feat = build_graph_and_extract_network_features(trans_train_val, accounts)
    account_features = account_features.merge(network_feat, on='ACCOUNT_ID', how='left')
    
    # 时序特征
    temporal_feat = extract_temporal_features(trans_train_val, accounts)
    account_features = account_features.merge(temporal_feat, on='ACCOUNT_ID', how='left')
    
    # 类别编码
    if 'COUNTRY' in account_features.columns:
        country_freq = accounts['COUNTRY'].value_counts(normalize=True).to_dict()
        account_features['country_freq'] = account_features['COUNTRY'].map(country_freq).fillna(0)
        account_features.drop('COUNTRY', axis=1, inplace=True)
    if 'ACCOUNT_TYPE' in account_features.columns:
        type_dummies = pd.get_dummies(account_features['ACCOUNT_TYPE'], prefix='type')
        account_features = pd.concat([account_features, type_dummies], axis=1)
        account_features.drop('ACCOUNT_TYPE', axis=1, inplace=True)
    if 'TX_BEHAVIOR_ID' in account_features.columns:
        behavior_freq = accounts['TX_BEHAVIOR_ID'].value_counts(normalize=True).to_dict()
        account_features['behavior_freq'] = account_features['TX_BEHAVIOR_ID'].map(behavior_freq).fillna(0)
        account_features.drop('TX_BEHAVIOR_ID', axis=1, inplace=True)
    
    exclude_cols = ['ACCOUNT_ID', 'IS_FRAUD']
    feature_cols = [col for col in account_features.columns if col not in exclude_cols]
    X = account_features[feature_cols].fillna(0)
    y = account_features['IS_FRAUD']
    print(f"特征矩阵形状: {X.shape}, 正样本比例: {y.mean():.4f}")
    return X, y, feature_cols, account_features[['ACCOUNT_ID']]

# ===================== 3. 阈值搜索（召回率优先）=====================
def find_threshold_by_recall(y_val, y_val_proba, recall_target):
    thresholds = np.linspace(0.01, 0.99, 200)
    best_th = 0.5
    best_precision = 0
    for th in thresholds:
        y_pred = (y_val_proba >= th).astype(int)
        rec = recall_score(y_val, y_pred)
        if rec >= recall_target:
            prec = precision_score(y_val, y_pred)
            if prec > best_precision:
                best_precision = prec
                best_th = th
    actual_recall = recall_score(y_val, (y_val_proba >= best_th).astype(int))
    print(f"召回率目标={recall_target:.0%} → 最佳阈值={best_th:.2f} (实际召回率={actual_recall:.4f}, 精确率={best_precision:.4f})")
    return best_th

# ===================== 4. 主流程 =====================
def run(data_dir, recall_target=0.80, rf_weight=0.5, xgb_weight=0.5):
    print(f"\n{'='*60}")
    print(f"AML Detection Model v10.0 (混合采样 + 召回率优先 + 集成)")
    print(f"数据集: {data_dir}")
    print(f"召回率目标: {recall_target:.0%}")
    print(f"随机森林权重: {rf_weight}, XGBoost权重: {xgb_weight}")
    print(f"{'='*60}\n")
    
    trans, accounts = load_data(data_dir)
    trans_train, trans_val, trans_test = split_by_time(trans, train_ratio=0.8, val_ratio=0.1)
    trans_train_val = pd.concat([trans_train, trans_val], axis=0)
    
    # 提取特征
    X_train_val, y_train_val, feature_cols, _ = extract_features(trans_train_val, accounts)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)
    print(f"训练集: {len(X_train)} (可疑: {y_train.sum()})")
    print(f"验证集: {len(X_val)} (可疑: {y_val.sum()})")
    
    # 混合采样
    print("\n应用 SMOTETomek 混合采样...")
    smt = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
    print(f"采样后训练集: {len(X_train_res)} (可疑: {y_train_res.sum()})")
    
    # 训练随机森林
    print("\n训练随机森林模型...")
    if USE_GPU_RF:
        rf = CumlRandomForest(random_state=42, n_estimators=200, max_depth=None, min_samples_split=2, 
                              class_weight='balanced_subsample')
        rf.fit(X_train_res, y_train_res)
        print("GPU 随机森林训练完成")
    else:
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=None, min_samples_split=2,
                                    class_weight='balanced_subsample', n_jobs=-1)
        rf.fit(X_train_res, y_train_res)
        print("CPU 随机森林训练完成")
    
    # 训练 XGBoost
    if XGB_AVAILABLE:
        print("\n训练 XGBoost 模型...")
        # 采样后正负样本已平衡，scale_pos_weight 设为 1
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            scale_pos_weight=1,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
        xgb_model.fit(X_train_res, y_train_res)
        print("XGBoost 训练完成")
    else:
        xgb_model = None
    
    # 验证集预测概率
    rf_val_proba = rf.predict_proba(X_val)[:, 1]
    if xgb_model is not None:
        xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]
        y_val_proba = rf_weight * rf_val_proba + xgb_weight * xgb_val_proba
    else:
        y_val_proba = rf_val_proba
    
    # 召回率优先阈值
    best_th = find_threshold_by_recall(y_val, y_val_proba, recall_target)
    
    # ========== 测试集特征构建 ==========
    print("\n构建测试集特征...")
    sender_stats_test = trans_test.groupby('SENDER_ACCOUNT_ID').agg(
        send_sum=('TX_AMOUNT', 'sum'), send_mean=('TX_AMOUNT', 'mean'), send_max=('TX_AMOUNT', 'max'),
        send_std=('TX_AMOUNT', 'std'), send_count=('TX_AMOUNT', 'count')
    ).reset_index().rename(columns={'SENDER_ACCOUNT_ID': 'ACCOUNT_ID'})
    receiver_stats_test = trans_test.groupby('RECEIVER_ACCOUNT_ID').agg(
        recv_sum=('TX_AMOUNT', 'sum'), recv_mean=('TX_AMOUNT', 'mean'), recv_max=('TX_AMOUNT', 'max'),
        recv_std=('TX_AMOUNT', 'std'), recv_count=('TX_AMOUNT', 'count')
    ).reset_index().rename(columns={'RECEIVER_ACCOUNT_ID': 'ACCOUNT_ID'})
    account_test = accounts.copy()
    account_test = account_test.merge(sender_stats_test, on='ACCOUNT_ID', how='left')
    account_test = account_test.merge(receiver_stats_test, on='ACCOUNT_ID', how='left')
    fill_cols = ['send_sum', 'send_mean', 'send_max', 'send_std', 'send_count',
                 'recv_sum', 'recv_mean', 'recv_max', 'recv_std', 'recv_count']
    account_test[fill_cols] = account_test[fill_cols].fillna(0)
    account_test['out_in_ratio'] = account_test['send_count'] / (account_test['recv_count'] + 1e-6)
    account_test['net_flow'] = account_test['send_sum'] - account_test['recv_sum']
    
    # 网络特征（复用训练图）
    G_train = nx.from_pandas_edgelist(trans_train_val, source='SENDER_ACCOUNT_ID', target='RECEIVER_ACCOUNT_ID', create_using=nx.DiGraph())
    nodes_test = list(account_test['ACCOUNT_ID'].values)
    out_degree_test = dict(G_train.out_degree(nodes_test))
    in_degree_test = dict(G_train.in_degree(nodes_test))
    pagerank_test = nx.pagerank(G_train, alpha=0.85, max_iter=100)
    clustering_test = nx.clustering(G_train.to_undirected())
    network_feat_test = pd.DataFrame({'ACCOUNT_ID': nodes_test})
    network_feat_test['out_degree'] = network_feat_test['ACCOUNT_ID'].map(out_degree_test).fillna(0)
    network_feat_test['in_degree'] = network_feat_test['ACCOUNT_ID'].map(in_degree_test).fillna(0)
    network_feat_test['total_degree'] = network_feat_test['out_degree'] + network_feat_test['in_degree']
    network_feat_test['pagerank'] = network_feat_test['ACCOUNT_ID'].map(pagerank_test).fillna(0)
    network_feat_test['clustering'] = network_feat_test['ACCOUNT_ID'].map(clustering_test).fillna(0)
    account_test = account_test.merge(network_feat_test, on='ACCOUNT_ID', how='left')
    
    # 时序特征（基于测试期交易）
    temporal_feat_test = extract_temporal_features(trans_test, accounts)
    account_test = account_test.merge(temporal_feat_test, on='ACCOUNT_ID', how='left')
    
    # 类别编码
    if 'COUNTRY' in account_test.columns:
        country_freq_test = account_test['COUNTRY'].value_counts(normalize=True).to_dict()
        account_test['country_freq'] = account_test['COUNTRY'].map(country_freq_test).fillna(0)
        account_test.drop('COUNTRY', axis=1, inplace=True)
    if 'ACCOUNT_TYPE' in account_test.columns:
        type_dummies_test = pd.get_dummies(account_test['ACCOUNT_TYPE'], prefix='type')
        account_test = pd.concat([account_test, type_dummies_test], axis=1)
        account_test.drop('ACCOUNT_TYPE', axis=1, inplace=True)
    if 'TX_BEHAVIOR_ID' in account_test.columns:
        behavior_freq_test = account_test['TX_BEHAVIOR_ID'].value_counts(normalize=True).to_dict()
        account_test['behavior_freq'] = account_test['TX_BEHAVIOR_ID'].map(behavior_freq_test).fillna(0)
        account_test.drop('TX_BEHAVIOR_ID', axis=1, inplace=True)
    
    X_test = account_test[feature_cols].fillna(0)
    y_test = account_test['IS_FRAUD']
    print(f"测试集大小: {len(X_test)} (可疑: {y_test.sum()})")
    
    # 测试集预测
    rf_test_proba = rf.predict_proba(X_test)[:, 1]
    if xgb_model is not None:
        xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]
        y_test_proba = rf_weight * rf_test_proba + xgb_weight * xgb_test_proba
    else:
        y_test_proba = rf_test_proba
    y_test_pred = (y_test_proba >= best_th).astype(int)
    
    # 静默账户过滤
    test_tx_count = trans_test.groupby('SENDER_ACCOUNT_ID').size().to_dict()
    silent_accounts = [acc for acc in account_test['ACCOUNT_ID'] if test_tx_count.get(acc, 0) == 0]
    for idx, acc in enumerate(account_test['ACCOUNT_ID']):
        if acc in silent_accounts:
            y_test_pred[idx] = 0
    y_test_pred = np.array(y_test_pred)
    print(f"\n后处理：剔除了 {len(silent_accounts)} 个静默账户（测试期内无交易）")
    
    print("\n" + "="*60)
    print("最终测试集评估结果（后处理后）")
    print("="*60)
    print(classification_report(y_test, y_test_pred, target_names=['正常', '可疑']))
    print(f"AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
    print(f"召回率: {recall_score(y_test, y_test_pred):.4f}")
    print(f"精确率: {precision_score(y_test, y_test_pred):.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    total = len(y_test)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['正常','可疑'], yticklabels=['正常','可疑'])
    plt.xlabel('预测'); plt.ylabel('真实'); plt.title(f'测试集混淆矩阵(阈值={best_th:.2f})(总样本={total})')
    plt.savefig('confusion_matrix_v10.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 特征重要性
    if not USE_GPU_RF:
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12,8))
        plt.title('随机森林特征重要性 (v10)')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance_v10.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("特征重要性图已保存: feature_importance_v10.png")
    
    # SHAP 分析
    if not USE_GPU_RF:
        print("\n计算 SHAP 值...")
        X_sample = X_test.sample(min(200, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)
        shap_values_class = shap_values[1] if isinstance(shap_values, list) else shap_values[:,:,1]
        shap.summary_plot(shap_values_class, X_sample, feature_names=feature_cols, show=False)
        plt.savefig('shap_summary_v10.png', dpi=150, bbox_inches='tight')
        plt.close()
        y_true_sample = y_test.loc[X_sample.index]
        pos = (y_true_sample == 1)
        if pos.any():
            idx = X_sample.index[pos][0]
            row_loc = X_sample.index.get_loc(idx)
            exp_val = explainer.expected_value
            base_val = float(exp_val[1]) if isinstance(exp_val, (list, np.ndarray)) and len(exp_val)>=2 else float(exp_val)
            exp = shap.Explanation(values=shap_values_class[row_loc],
                                   base_values=base_val,
                                   data=X_sample.iloc[row_loc, :].values,
                                   feature_names=feature_cols)
            shap.waterfall_plot(exp, show=False)
            plt.savefig('shap_waterfall_v10.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("SHAP 瀑布图已保存: shap_waterfall_v10.png")
        else:
            print("样本子集中无真实可疑账户，跳过瀑布图")
    
    # 输出可疑账户名单
    test_account_ids = account_test['ACCOUNT_ID'].values
    test_results = pd.DataFrame({
        'ACCOUNT_ID': test_account_ids,
        'FRAUD_PROBABILITY': y_test_proba,
        'PREDICTION': y_test_pred,
        'TRUE_LABEL': y_test.values
    })
    suspicious_list = test_results[test_results['PREDICTION'] == 1].copy()
    suspicious_list = suspicious_list.sort_values('FRAUD_PROBABILITY', ascending=False)
    suspicious_list.to_csv('suspicious_accounts_v10.csv', index=False)
    print(f"\n可疑账户名单已保存: suspicious_accounts_v10.csv")
    print(f"共识别出 {len(suspicious_list)} 个可疑账户 (其中真实可疑 {suspicious_list['TRUE_LABEL'].sum()} 个)")
    
    return rf, xgb_model, best_th

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Detection Model v10.0")
    parser.add_argument('--data_dir', type=str, required=True, help='数据文件夹路径')
    parser.add_argument('--recall_target', type=float, default=0.80, help='召回率目标 (0-1)，默认0.80')
    parser.add_argument('--rf_weight', type=float, default=0.5, help='随机森林权重（0~1）')
    parser.add_argument('--xgb_weight', type=float, default=0.5, help='XGBoost 权重（0~1）')
    args = parser.parse_args()
    total = args.rf_weight + args.xgb_weight
    rf_w = args.rf_weight / total
    xgb_w = args.xgb_weight / total
    run(args.data_dir, args.recall_target, rf_w, xgb_w)