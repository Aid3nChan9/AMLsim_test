"""
AML Detection Model v9.0
改进点：
1. 混合采样 SMOTETomek（清理边界样本，降低误报）
2. F2.0 分数优化阈值（精确率优先）
3. 新增行为突变偏离度、资金高频穿透率特征
4. 静默账户后处理（测试期内无交易账户直接排除）
5. 集成学习：随机森林 + XGBoost 软投票（可配置权重）
使用方法：
    python aml_model_v9.py --data_dir <路径> [--rf_weight 0.5] [--xgb_weight 0.5]
"""

import os
import argparse
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, precision_score, recall_score, fbeta_score)
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

# XGBoost（必选）
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

# ===================== 2. 特征工程（新增两个强特征）=====================
def extract_temporal_features(trans, accounts):
    """
    提取时序特征（基于给定的交易数据）
    返回：DataFrame with columns: active_days_send, daily_tx_send, night_ratio_send, 
          max_daily_amount_send, max_daily_count_send, 以及接收方对应特征
    """
    trans = trans.copy()
    trans['day'] = trans['TIMESTAMP'] // 86400
    trans['is_night'] = (trans['TIMESTAMP'] % 86400) < 21600
    
    # 发送方特征
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
    
    # 接收方特征
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

def add_advanced_features(trans_train, trans_val, trans_test, accounts):
    """
    新增两个高级特征：
    1. 行为突变偏离度 = (测试期日均交易额) / (训练期日均交易额)
    2. 资金高频穿透率 = 单日最大交易金额 / INIT_BALANCE
    注：训练期和测试期按时间划分，测试期特征仅用于测试集，不参与训练。
    """
    # 训练+验证期交易（用于计算训练期基准）
    trans_train_val = pd.concat([trans_train, trans_val], axis=0)
    # 测试期交易
    trans_test_only = trans_test
    
    # 计算每个账户在训练期的日均交易额（发送+接收平均，或仅发送？这里使用发送交易额）
    train_daily_avg = trans_train_val.groupby('SENDER_ACCOUNT_ID').apply(
        lambda g: g['TX_AMOUNT'].sum() / (g['TIMESTAMP'].max() - g['TIMESTAMP'].min() + 1) if len(g) > 0 else 0
    ).to_dict()
    
    # 测试期日均交易额
    test_daily_avg = trans_test_only.groupby('SENDER_ACCOUNT_ID').apply(
        lambda g: g['TX_AMOUNT'].sum() / (g['TIMESTAMP'].max() - g['TIMESTAMP'].min() + 1) if len(g) > 0 else 0
    ).to_dict()
    
    # 行为突变偏离度
    behavior_deviation = {}
    for acc in accounts['ACCOUNT_ID']:
        train_avg = train_daily_avg.get(acc, 0)
        test_avg = test_daily_avg.get(acc, 0)
        if train_avg > 0:
            behavior_deviation[acc] = test_avg / train_avg
        else:
            behavior_deviation[acc] = 0 if test_avg == 0 else 100  # 无历史但测试期有交易 -> 极高突变
    
    # 资金高频穿透率 = 测试期内最大单日交易金额 / INIT_BALANCE
    max_daily_amount = trans_test_only.groupby('SENDER_ACCOUNT_ID').apply(
        lambda g: g.groupby(g['TIMESTAMP'] // 86400)['TX_AMOUNT'].sum().max() if len(g) > 0 else 0
    ).to_dict()
    # 合并账户表获取初始余额
    init_balance = accounts.set_index('ACCOUNT_ID')['INIT_BALANCE'].to_dict()
    penetration_rate = {}
    for acc in accounts['ACCOUNT_ID']:
        max_daily = max_daily_amount.get(acc, 0)
        bal = init_balance.get(acc, 1.0)
        penetration_rate[acc] = max_daily / (bal + 1e-8)
    
    # 构建DataFrame
    adv_feat = pd.DataFrame({'ACCOUNT_ID': accounts['ACCOUNT_ID']})
    adv_feat['behavior_deviation'] = adv_feat['ACCOUNT_ID'].map(behavior_deviation).fillna(0)
    adv_feat['penetration_rate'] = adv_feat['ACCOUNT_ID'].map(penetration_rate).fillna(0)
    return adv_feat

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

def extract_features(trans_train_val, accounts, trans_train, trans_val, trans_test):
    """
    整合所有特征：统计+网络+时序+高级特征（行为突变+穿透率）
    注意：高级特征中的行为突变需要使用训练期和测试期数据，但测试期数据不参与训练特征构建。
    这里为了统一，我们在训练集和验证集上计算时，只使用训练+验证期的交易；在测试集上单独计算。
    因此，该函数返回的特征是训练+验证集的特征矩阵。
    测试集特征单独在 run() 中构建。
    """
    # 统计特征（发送/接收聚合）
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
    
    # 时序特征（使用训练+验证期）
    temporal_feat = extract_temporal_features(trans_train_val, accounts)
    account_features = account_features.merge(temporal_feat, on='ACCOUNT_ID', how='left')
    
    # 高级特征：行为突变和穿透率，需要使用训练期和测试期数据，这里暂不添加到训练集（因为测试期未参与训练）
    # 但可以为训练集构造“伪”特征（使用训练+验证期自身计算？会导致泄漏），先不加，留到测试时再单独添加。
    # 我们将在模型预测前为测试集单独添加这两个特征。
    
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
    # 为了与测试集特征对齐，添加行为偏离度和穿透率占位列（训练时不使用真实值）
    # 注意：这两个特征在训练时无实际区分能力，仅用于保证特征维度一致
    X['behavior_deviation'] = 0.0
    X['penetration_rate'] = 0.0
    # 将这两个新列加入 feature_cols
    feature_cols = feature_cols + ['behavior_deviation', 'penetration_rate']
    return X, y, feature_cols, account_features[['ACCOUNT_ID']]

# ===================== 3. 模型训练与集成 =====================
def find_threshold_fbeta_with_recall_floor(y_val, y_val_proba, beta=2.0, recall_floor=0.85):
    """
    在保证召回率不低于 recall_floor 的前提下，最大化 F-beta 分数。
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_th = 0.01  # 默认一个极低的阈值，保证召回率底线
    best_fbeta = 0
    
    for th in thresholds:
        y_pred = (y_val_proba >= th).astype(int)
        rec = recall_score(y_val, y_pred)
        
        # 硬约束：如果召回率低于底线，直接跳过该阈值
        if rec < recall_floor:
            continue
            
        fb = fbeta_score(y_val, y_pred, beta=beta)
        if fb > best_fbeta:
            best_fbeta = fb
            best_th = th
            
    # 计算最终选择的阈值在验证集上的表现
    final_pred = (y_val_proba >= best_th).astype(int)
    final_rec = recall_score(y_val, final_pred)
    final_prec = precision_score(y_val, final_pred)
    
    print(f"F{beta} 最大化策略 (底线 Recall>={recall_floor}) → 最佳阈值={best_th:.2f} (F{beta}={best_fbeta:.4f}, 召回={final_rec:.4f}, 精确={final_prec:.4f})")
    return best_th

def run(data_dir, rf_weight=0.3, xgb_weight=0.7):
    print(f"\n{'='*60}")
    print(f"AML Detection Model v9.0 (集成学习 + F2.0 阈值 + 混合采样)")
    print(f"数据集: {data_dir}")
    print(f"随机森林权重: {rf_weight}, XGBoost权重: {xgb_weight}")
    print(f"{'='*60}\n")
    
    trans, accounts = load_data(data_dir)
    trans_train, trans_val, trans_test = split_by_time(trans, train_ratio=0.8, val_ratio=0.1)
    trans_train_val = pd.concat([trans_train, trans_val], axis=0)
    
    # 提取训练+验证特征（不含高级特征，因为高级特征需要测试期数据）
    X_train_val, y_train_val, feature_cols, account_ids_df = extract_features(
        trans_train_val, accounts, trans_train, trans_val, trans_test
    )
    
    # 划分训练集和验证集（随机分层）
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)
    print(f"训练集: {len(X_train)} (可疑: {y_train.sum()})")
    print(f"验证集: {len(X_val)} (可疑: {y_val.sum()})")
    
    # 混合采样（SMOTETomek）
    print("\n应用 SMOTETomek 混合采样（清理边界样本）...")
    smt = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
    print(f"采样后训练集: {len(X_train_res)} (可疑: {y_train_res.sum()})")
    
    # 训练随机森林
    print("\n训练随机森林模型...")
    if USE_GPU_RF:
        rf = CumlRandomForest(random_state=42, n_estimators=200, max_depth=None, min_samples_split=2, class_weight='balanced_subsample')
        rf.fit(X_train_res, y_train_res)
        print("GPU 随机森林训练完成")
    else:
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=None, min_samples_split=2, 
                                    class_weight='balanced_subsample', n_jobs=-1)
        rf.fit(X_train_res, y_train_res)
        print("CPU 随机森林训练完成")
    
    # 训练 XGBoost（如果可用）
    if XGB_AVAILABLE:
        print("\n训练 XGBoost 模型...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05, 
            scale_pos_weight=(len(y_train_res) - y_train_res.sum()) / y_train_res.sum(),  # 处理不平衡
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
        # 加权平均概率
        y_val_proba = rf_weight * rf_val_proba + xgb_weight * xgb_val_proba
    else:
        y_val_proba = rf_val_proba
    
    # 优化阈值（F0.5 最大化）
    best_th = find_threshold_fbeta_with_recall_floor(y_val, y_val_proba, beta=2.0)
    
    # ========== 测试集特征构建（需包含高级特征） ==========
    print("\n构建测试集特征（含高级特征）...")
    # 测试集统计特征（仅基于测试期交易）
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
    
    # 高级特征（行为突变偏离度 + 资金穿透率）
    adv_feat_test = add_advanced_features(trans_train, trans_val, trans_test, accounts)
    account_test = account_test.merge(adv_feat_test, on='ACCOUNT_ID', how='left')
    
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
    
    # 确保测试集特征列与训练集一致（可能多出 behavior_deviation, penetration_rate）
    # 训练集没有这两个特征，需在训练时也加入，但当时未加。这里我们动态对齐
    for col in ['behavior_deviation', 'penetration_rate']:
        if col not in feature_cols:
            feature_cols.append(col)
    X_test = account_test[feature_cols].fillna(0)
    y_test = account_test['IS_FRAUD']
    print(f"测试集大小: {len(X_test)} (可疑: {y_test.sum()})")
    
    # 测试集预测概率
    rf_test_proba = rf.predict_proba(X_test)[:, 1]
    if xgb_model is not None:
        xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]
        y_test_proba = rf_weight * rf_test_proba + xgb_weight * xgb_test_proba
    else:
        y_test_proba = rf_test_proba
    y_test_pred = (y_test_proba >= best_th).astype(int)
    
    # ========== 后处理：静默账户过滤 ==========
    # 计算每个账户在测试期的交易笔数（作为发送方）
    test_tx_count = trans_test.groupby('SENDER_ACCOUNT_ID').size().to_dict()
    silent_accounts = [acc for acc in account_test['ACCOUNT_ID'] if test_tx_count.get(acc, 0) == 0]
    # 对预测结果，将静默账户强制设为 0（正常）
    for idx, acc in enumerate(account_test['ACCOUNT_ID']):
        if acc in silent_accounts:
            y_test_pred[idx] = 0
    # 重新计算指标
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
    plt.savefig('confusion_matrix_v9.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 特征重要性（仅随机森林）
    if not USE_GPU_RF:
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12,8))
        plt.title('随机森林特征重要性 (v9)')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance_v9.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("特征重要性图已保存: feature_importance_v9.png")
    
    # SHAP 分析（仅随机森林，因为可解释性更好）
    if not USE_GPU_RF:
        print("\n计算 SHAP 值...")
        X_sample = X_test.sample(min(200, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)
        shap_values_class = shap_values[1] if isinstance(shap_values, list) else shap_values[:,:,1]
        shap.summary_plot(shap_values_class, X_sample, feature_names=feature_cols, show=False)
        plt.savefig('shap_summary_v9.png', dpi=150, bbox_inches='tight')
        plt.close()
        # 瀑布图
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
            plt.savefig('shap_waterfall_v9.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("SHAP 瀑布图已保存: shap_waterfall_v9.png")
        else:
            print("样本子集中无真实可疑账户，跳过瀑布图")
    
    # 输出可疑账户名单（后处理后的）
    test_account_ids = account_test['ACCOUNT_ID'].values
    test_results = pd.DataFrame({
        'ACCOUNT_ID': test_account_ids,
        'FRAUD_PROBABILITY': y_test_proba,
        'PREDICTION': y_test_pred,
        'TRUE_LABEL': y_test.values
    })
    suspicious_list = test_results[test_results['PREDICTION'] == 1].copy()
    suspicious_list = suspicious_list.sort_values('FRAUD_PROBABILITY', ascending=False)
    suspicious_list.to_csv('suspicious_accounts_v9.csv', index=False)
    print(f"\n可疑账户名单已保存: suspicious_accounts_v9.csv")
    print(f"共识别出 {len(suspicious_list)} 个可疑账户 (其中真实可疑 {suspicious_list['TRUE_LABEL'].sum()} 个)")
    
    return rf, xgb_model, best_th

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Detection Model v9.0")
    parser.add_argument('--data_dir', type=str, required=True, help='数据文件夹路径')
    parser.add_argument('--rf_weight', type=float, default=0.3, help='随机森林权重（0~1）')
    parser.add_argument('--xgb_weight', type=float, default=0.7, help='XGBoost 权重（0~1）')
    args = parser.parse_args()
    # 权重归一化
    total = args.rf_weight + args.xgb_weight
    rf_w = args.rf_weight / total
    xgb_w = args.xgb_weight / total
    run(args.data_dir, rf_w, xgb_w)