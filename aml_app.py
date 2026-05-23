"""
AML 反洗钱检测系统 - 桌面应用程序
基于 aml_model_v11.py 内核，tkinter GUI 界面
打包为独立 .exe 运行
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime
import traceback

# 获取程序运行目录（支持 exe 打包后路径）
if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_score, recall_score, fbeta_score)
from imblearn.combine import SMOTETomek
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

# ---------- XGBoost / cuML ----------
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from cuml.ensemble import RandomForestClassifier as CumlRF
    USE_GPU_RF = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    USE_GPU_RF = False

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 核心模型函数（来自 v11） ====================

def load_data(data_dir):
    trans = pd.read_csv(os.path.join(data_dir, 'transactions.csv'))
    accounts = pd.read_csv(os.path.join(data_dir, 'accounts.csv'))
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
    return trans_train, trans_val, trans_test

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
    for acc in set(night_amount_send) | set(total_amount_send):
        night_ratio_send[acc] = night_amount_send.get(acc, 0.0) / (total_amount_send.get(acc, 0.0) + 1e-8)
    daily_amount_max_send = sender_group.apply(lambda x: x.groupby('day')['TX_AMOUNT'].sum().max()).to_dict()
    daily_count_max_send = sender_group.apply(lambda x: x.groupby('day').size().max()).to_dict()
    receiver_group = trans.groupby('RECEIVER_ACCOUNT_ID')
    active_days_recv = receiver_group['day'].nunique().to_dict()
    daily_tx_recv = (receiver_group.size() / receiver_group['day'].nunique()).to_dict()
    night_amount_recv = trans[trans['is_night']].groupby('RECEIVER_ACCOUNT_ID')['TX_AMOUNT'].sum().to_dict()
    total_amount_recv = receiver_group['TX_AMOUNT'].sum().to_dict()
    night_ratio_recv = {}
    for acc in set(night_amount_recv) | set(total_amount_recv):
        night_ratio_recv[acc] = night_amount_recv.get(acc, 0.0) / (total_amount_recv.get(acc, 0.0) + 1e-8)
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
    G = nx.from_pandas_edgelist(trans_train_val, source='SENDER_ACCOUNT_ID',
                                target='RECEIVER_ACCOUNT_ID', create_using=nx.DiGraph())
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
    sender_stats = trans_train_val.groupby('SENDER_ACCOUNT_ID').agg(
        send_sum=('TX_AMOUNT', 'sum'), send_mean=('TX_AMOUNT', 'mean'),
        send_max=('TX_AMOUNT', 'max'), send_std=('TX_AMOUNT', 'std'),
        send_count=('TX_AMOUNT', 'count')
    ).reset_index().rename(columns={'SENDER_ACCOUNT_ID': 'ACCOUNT_ID'})
    receiver_stats = trans_train_val.groupby('RECEIVER_ACCOUNT_ID').agg(
        recv_sum=('TX_AMOUNT', 'sum'), recv_mean=('TX_AMOUNT', 'mean'),
        recv_max=('TX_AMOUNT', 'max'), recv_std=('TX_AMOUNT', 'std'),
        recv_count=('TX_AMOUNT', 'count')
    ).reset_index().rename(columns={'RECEIVER_ACCOUNT_ID': 'ACCOUNT_ID'})
    account_features = accounts[['ACCOUNT_ID', 'IS_FRAUD', 'INIT_BALANCE',
                                  'COUNTRY', 'ACCOUNT_TYPE', 'TX_BEHAVIOR_ID']].copy()
    account_features = account_features.merge(sender_stats, on='ACCOUNT_ID', how='left')
    account_features = account_features.merge(receiver_stats, on='ACCOUNT_ID', how='left')
    fill_cols = ['send_sum', 'send_mean', 'send_max', 'send_std', 'send_count',
                 'recv_sum', 'recv_mean', 'recv_max', 'recv_std', 'recv_count']
    account_features[fill_cols] = account_features[fill_cols].fillna(0)
    account_features['out_in_ratio'] = account_features['send_count'] / (account_features['recv_count'] + 1e-6)
    account_features['net_flow'] = account_features['send_sum'] - account_features['recv_sum']
    network_feat = build_graph_and_extract_network_features(trans_train_val, accounts)
    account_features = account_features.merge(network_feat, on='ACCOUNT_ID', how='left')
    temporal_feat = extract_temporal_features(trans_train_val, accounts)
    account_features = account_features.merge(temporal_feat, on='ACCOUNT_ID', how='left')

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
    return X, y, feature_cols, account_features

def find_threshold_fbeta(y_val, y_val_proba, beta=1.5):
    thresholds = np.linspace(0.01, 0.99, 200)
    best_th, best_fbeta = 0.5, 0
    for th in thresholds:
        y_pred = (y_val_proba >= th).astype(int)
        fb = fbeta_score(y_val, y_pred, beta=beta)
        if fb > best_fbeta:
            best_fbeta, best_th = fb, th
    return best_th

# ==================== 模型训练器（在线程中运行） ====================

class ModelRunner:
    """在后台线程中运行模型训练"""

    def __init__(self, data_dir, rf_weight, xgb_weight, callback):
        self.data_dir = data_dir
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        self.callback = callback
        self.result = None

    def log(self, msg):
        self.callback('log', msg)

    def progress(self, pct, label=""):
        self.callback('progress', {'pct': pct, 'label': label})

    def run(self):
        try:
            self.log("=" * 50)
            self.log("AML 反洗钱检测系统 v11 - 开始运行")
            self.log("=" * 50)

            # 1. 加载数据
            self.log(f"数据目录: {self.data_dir}")
            self.progress(5, "加载数据...")
            trans, accounts = load_data(self.data_dir)
            self.log(f"交易数据: {len(trans):,} 笔  |  账户: {len(accounts):,} 个")
            self.log(f"可疑账户占比: {accounts['IS_FRAUD'].mean():.2%}")

            # 2. 时间划分
            self.progress(10, "时间划分...")
            trans_train, trans_val, trans_test = split_by_time(trans)
            trans_train_val = pd.concat([trans_train, trans_val])
            self.log(f"训练集: {len(trans_train):,}  验证集: {len(trans_val):,}  测试集: {len(trans_test):,}")

            # 3. 特征提取
            self.progress(20, "提取特征...")
            X_train_val, y_train_val, feature_cols, _ = extract_features(trans_train_val, accounts)
            self.log(f"特征维度: {X_train_val.shape}")

            # 4. 划分训练/验证
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)

            # 5. SMOTETomek
            self.progress(40, "SMOTETomek 混合采样...")
            smt = SMOTETomek(random_state=42)
            X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
            self.log(f"采样后: {len(X_train_res)} 样本 (可疑: {y_train_res.sum()})")

            # 6. 训练随机森林
            self.progress(50, "训练随机森林...")
            if USE_GPU_RF:
                rf = CumlRF(random_state=42, n_estimators=200, max_depth=None, min_samples_split=2,
                            class_weight='balanced_subsample')
                rf.fit(X_train_res.values, y_train_res.values)
            else:
                rf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=None,
                                            min_samples_split=2, class_weight='balanced_subsample', n_jobs=-1)
                rf.fit(X_train_res, y_train_res)
            self.log("随机森林训练完成")

            # 7. 训练 XGBoost
            if XGB_AVAILABLE:
                self.progress(60, "训练 XGBoost...")
                xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                               random_state=42, eval_metric='logloss')
                xgb_model.fit(X_train_res, y_train_res)
                self.log("XGBoost 训练完成")
            else:
                xgb_model = None

            # 8. 验证集预测与阈值优化
            self.progress(70, "阈值优化...")
            rf_val_proba = rf.predict_proba(X_val)[:, 1]
            if xgb_model is not None:
                xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]
                y_val_proba = self.rf_weight * rf_val_proba + self.xgb_weight * xgb_val_proba
            else:
                y_val_proba = rf_val_proba
            best_th = find_threshold_fbeta(y_val, y_val_proba, beta=1.5)

            # 9. 测试集预测
            self.progress(80, "构建测试集特征...")
            test_result = self._run_test(trans_train_val, trans_test, accounts, feature_cols,
                                          rf, xgb_model, best_th)
            self.progress(95, "生成可视化...")
            self._generate_visuals(**test_result, best_th=best_th)

            self.progress(100, "完成！")
            self.callback('done', test_result)

        except Exception as e:
            self.log(f"错误: {traceback.format_exc()}")
            self.callback('error', str(e))

    def _run_test(self, trans_train_val, trans_test, accounts, feature_cols, rf, xgb_model, best_th):
        """测试集评估"""
        sender_stats_t = trans_test.groupby('SENDER_ACCOUNT_ID').agg(
            send_sum=('TX_AMOUNT', 'sum'), send_mean=('TX_AMOUNT', 'mean'),
            send_max=('TX_AMOUNT', 'max'), send_std=('TX_AMOUNT', 'std'),
            send_count=('TX_AMOUNT', 'count')).reset_index().rename(columns={'SENDER_ACCOUNT_ID': 'ACCOUNT_ID'})
        receiver_stats_t = trans_test.groupby('RECEIVER_ACCOUNT_ID').agg(
            recv_sum=('TX_AMOUNT', 'sum'), recv_mean=('TX_AMOUNT', 'mean'),
            recv_max=('TX_AMOUNT', 'max'), recv_std=('TX_AMOUNT', 'std'),
            recv_count=('TX_AMOUNT', 'count')).reset_index().rename(columns={'RECEIVER_ACCOUNT_ID': 'ACCOUNT_ID'})
        account_test = accounts.copy()
        account_test = account_test.merge(sender_stats_t, on='ACCOUNT_ID', how='left')
        account_test = account_test.merge(receiver_stats_t, on='ACCOUNT_ID', how='left')
        fill_cols = ['send_sum', 'send_mean', 'send_max', 'send_std', 'send_count',
                     'recv_sum', 'recv_mean', 'recv_max', 'recv_std', 'recv_count']
        account_test[fill_cols] = account_test[fill_cols].fillna(0)
        account_test['out_in_ratio'] = account_test['send_count'] / (account_test['recv_count'] + 1e-6)
        account_test['net_flow'] = account_test['send_sum'] - account_test['recv_sum']

        G_train = nx.from_pandas_edgelist(trans_train_val, source='SENDER_ACCOUNT_ID',
                                          target='RECEIVER_ACCOUNT_ID', create_using=nx.DiGraph())
        nodes_test = list(account_test['ACCOUNT_ID'].values)
        network_feat_test = pd.DataFrame({'ACCOUNT_ID': nodes_test})
        network_feat_test['out_degree'] = network_feat_test['ACCOUNT_ID'].map(dict(G_train.out_degree(nodes_test))).fillna(0)
        network_feat_test['in_degree'] = network_feat_test['ACCOUNT_ID'].map(dict(G_train.in_degree(nodes_test))).fillna(0)
        network_feat_test['total_degree'] = network_feat_test['out_degree'] + network_feat_test['in_degree']
        pagerank_t = nx.pagerank(G_train, alpha=0.85, max_iter=100)
        clustering_t = nx.clustering(G_train.to_undirected())
        network_feat_test['pagerank'] = network_feat_test['ACCOUNT_ID'].map(pagerank_t).fillna(0)
        network_feat_test['clustering'] = network_feat_test['ACCOUNT_ID'].map(clustering_t).fillna(0)
        account_test = account_test.merge(network_feat_test, on='ACCOUNT_ID', how='left')

        temporal_feat_test = extract_temporal_features(trans_test, accounts)
        account_test = account_test.merge(temporal_feat_test, on='ACCOUNT_ID', how='left')

        if 'COUNTRY' in account_test.columns:
            account_test['country_freq'] = account_test['COUNTRY'].map(
                account_test['COUNTRY'].value_counts(normalize=True).to_dict()).fillna(0)
            account_test.drop('COUNTRY', axis=1, inplace=True)
        if 'ACCOUNT_TYPE' in account_test.columns:
            type_dummies = pd.get_dummies(account_test['ACCOUNT_TYPE'], prefix='type')
            account_test = pd.concat([account_test, type_dummies], axis=1)
            account_test.drop('ACCOUNT_TYPE', axis=1, inplace=True)
        if 'TX_BEHAVIOR_ID' in account_test.columns:
            account_test['behavior_freq'] = account_test['TX_BEHAVIOR_ID'].map(
                account_test['TX_BEHAVIOR_ID'].value_counts(normalize=True).to_dict()).fillna(0)
            account_test.drop('TX_BEHAVIOR_ID', axis=1, inplace=True)

        X_test = account_test[feature_cols].fillna(0)
        y_test = account_test['IS_FRAUD']

        rf_test_proba = rf.predict_proba(X_test)[:, 1]
        if xgb_model is not None:
            xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]
            y_test_proba = self.rf_weight * rf_test_proba + self.xgb_weight * xgb_test_proba
        else:
            y_test_proba = rf_test_proba
        y_test_pred = (y_test_proba >= best_th).astype(int)

        return {
            'X_test': X_test, 'y_test': y_test, 'y_test_proba': y_test_proba,
            'y_test_pred': y_test_pred, 'feature_cols': feature_cols,
            'rf': rf, 'xgb_model': xgb_model, 'best_th': best_th,
            'account_test': account_test
        }

    def _generate_visuals(self, X_test, y_test, y_test_proba, y_test_pred,
                           feature_cols, rf, xgb_model, best_th, account_test, **kwargs):
        output_dir = os.path.join(APP_DIR, 'output')
        os.makedirs(output_dir, exist_ok=True)

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['正常', '可疑'], yticklabels=['正常', '可疑'])
        plt.xlabel('预测'); plt.ylabel('真实'); plt.title(f'混淆矩阵 (阈值={best_th:.2f})')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 特征重要性
        if not USE_GPU_RF:
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(12, 8))
            plt.title('特征重要性 Top 20')
            top_n = min(20, len(indices))
            plt.barh(range(top_n), importances[indices[:top_n]], align='center')
            plt.yticks(range(top_n), [feature_cols[i] for i in indices[:top_n]])
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
            plt.close()

        # SHAP 汇总图
        if not USE_GPU_RF:
            try:
                X_sample = X_test.sample(min(200, len(X_test)), random_state=42)
                explainer = shap.TreeExplainer(rf)
                shap_values = explainer.shap_values(X_sample)
                sv = shap_values[1] if isinstance(shap_values, list) else shap_values[:,:,1]
                shap.summary_plot(sv, X_sample, feature_names=feature_cols, show=False)
                plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=150, bbox_inches='tight')
                plt.close()
            except:
                pass

        # 可疑账户名单
        test_results = pd.DataFrame({
            'ACCOUNT_ID': account_test['ACCOUNT_ID'].values,
            'FRAUD_PROBABILITY': y_test_proba,
            'PREDICTION': y_test_pred,
            'TRUE_LABEL': y_test.values
        })
        suspicious = test_results[test_results['PREDICTION'] == 1].sort_values('FRAUD_PROBABILITY', ascending=False)
        suspicious.to_csv(os.path.join(output_dir, 'suspicious_accounts.csv'), index=False, encoding='utf-8-sig')
        self.result = {
            'output_dir': output_dir,
            'auc': roc_auc_score(y_test, y_test_proba),
            'recall': recall_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'report': classification_report(y_test, y_test_pred, target_names=['正常', '可疑']),
            'suspicious_count': len(suspicious),
            'suspicious_true': int(suspicious['TRUE_LABEL'].sum()),
            'total_test': len(y_test),
            'best_th': best_th
        }

# ==================== GUI 界面 ====================

class AMLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AML 反洗钱检测系统 v11")
        self.root.geometry("950x720")
        self.root.minsize(800, 600)
        self.data_dir = tk.StringVar()
        self.rf_weight = tk.DoubleVar(value=0.5)
        self.xgb_weight = tk.DoubleVar(value=0.5)
        self.runner_thread = None
        self._build_ui()

    def _build_ui(self):
        # 顶部标题
        header = tk.Frame(self.root, bg="#1a5276", height=60)
        header.pack(fill=tk.X)
        tk.Label(header, text="AML 反洗钱检测系统 v11", font=("Microsoft YaHei", 18, "bold"),
                 fg="white", bg="#1a5276").pack(pady=12)

        # 主内容区
        main = tk.Frame(self.root, padx=20, pady=10)
        main.pack(fill=tk.BOTH, expand=True)

        # 参数配置区
        config_frame = tk.LabelFrame(main, text="参数配置", font=("Microsoft YaHei", 11), padx=10, pady=10)
        config_frame.pack(fill=tk.X, pady=(0, 10))

        row1 = tk.Frame(config_frame)
        row1.pack(fill=tk.X, pady=5)
        tk.Label(row1, text="数据目录:", font=("Microsoft YaHei", 10), width=10).pack(side=tk.LEFT)
        tk.Entry(row1, textvariable=self.data_dir, font=("Consolas", 9), width=60).pack(side=tk.LEFT, padx=5)
        tk.Button(row1, text="浏览...", command=self._browse_dir, width=8).pack(side=tk.LEFT)

        row2 = tk.Frame(config_frame)
        row2.pack(fill=tk.X, pady=5)
        tk.Label(row2, text="RF 权重:", font=("Microsoft YaHei", 10), width=10).pack(side=tk.LEFT)
        tk.Scale(row2, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=200,
                 variable=self.rf_weight).pack(side=tk.LEFT, padx=5)
        tk.Label(row2, text="XGB 权重:", font=("Microsoft YaHei", 10), width=10).pack(side=tk.LEFT)
        tk.Scale(row2, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=200,
                 variable=self.xgb_weight).pack(side=tk.LEFT, padx=5)
        tk.Label(row2, text="(自动归一化)", font=("Microsoft YaHei", 8), fg="gray").pack(side=tk.LEFT)

        # 操作按钮
        btn_frame = tk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=5)
        self.run_btn = tk.Button(btn_frame, text="▶  开始训练", font=("Microsoft YaHei", 12, "bold"),
                                  bg="#27ae60", fg="white", width=15, height=1, command=self._start_training)
        self.run_btn.pack(side=tk.LEFT, padx=5)
        self.open_btn = tk.Button(btn_frame, text="📂 打开结果目录", font=("Microsoft YaHei", 10),
                                   state=tk.DISABLED, command=self._open_output)
        self.open_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = tk.Button(btn_frame, text="■ 停止", font=("Microsoft YaHei", 10),
                                   fg="red", state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)

        self.progress_label = tk.Label(main, text="就绪，请选择数据集并开始训练",
                                        font=("Microsoft YaHei", 9), fg="gray")
        self.progress_label.pack(anchor=tk.W)

        # 主内容区（左右分栏）
        content = tk.Frame(main)
        content.pack(fill=tk.BOTH, expand=True, pady=5)

        # 左侧：日志
        left_frame = tk.LabelFrame(content, text="运行日志", font=("Microsoft YaHei", 10))
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.log_text = scrolledtext.ScrolledText(left_frame, font=("Consolas", 9), wrap=tk.WORD,
                                                   bg="#1e1e1e", fg="#d4d4d4")
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 右侧：结果
        right_frame = tk.LabelFrame(content, text="评估结果", font=("Microsoft YaHei", 10))
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.result_text = scrolledtext.ScrolledText(right_frame, font=("Consolas", 10), wrap=tk.WORD,
                                                      bg="#f8f9fa", height=15)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # 底部状态栏
        self.status_bar = tk.Label(self.root, text="就绪", font=("Microsoft YaHei", 8),
                                    bg="#e9ecef", anchor=tk.W, padx=10)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _browse_dir(self):
        path = filedialog.askdirectory(title="选择包含 transactions.csv 和 accounts.csv 的目录")
        if path:
            self.data_dir.set(path)
            self._log(f"已选择目录: {path}")

    def _log(self, msg):
        self.root.after(0, lambda: self._append_log(msg))

    def _append_log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)

    def _start_training(self):
        data_dir = self.data_dir.get()
        if not data_dir:
            messagebox.showwarning("警告", "请先选择数据目录！")
            return
        trans_path = os.path.join(data_dir, 'transactions.csv')
        acc_path = os.path.join(data_dir, 'accounts.csv')
        if not os.path.exists(trans_path) or not os.path.exists(acc_path):
            messagebox.showerror("错误", f"在所选目录中找不到 transactions.csv 或 accounts.csv！\n\n目录: {data_dir}")
            return

        self.log_text.delete('1.0', tk.END)
        self.result_text.delete('1.0', tk.END)
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.open_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)

        total = self.rf_weight.get() + self.xgb_weight.get()
        rf_w = self.rf_weight.get() / total if total > 0 else 0.5
        xgb_w = self.xgb_weight.get() / total if total > 0 else 0.5

        runner = ModelRunner(data_dir, rf_w, xgb_w, self._on_callback)
        self.runner_thread = threading.Thread(target=runner.run, daemon=True)
        self.runner_thread.start()

    def _on_callback(self, event_type, data):
        if event_type == 'log':
            self._log(data)
        elif event_type == 'progress':
            self.root.after(0, lambda: self._update_progress(data['pct'], data.get('label', '')))
        elif event_type == 'done':
            self.root.after(0, lambda: self._on_done(data))
        elif event_type == 'error':
            self.root.after(0, lambda: self._on_error(data))

    def _update_progress(self, pct, label):
        self.progress_var.set(pct)
        self.progress_label.config(text=f"{label} ({pct:.0f}%)")
        self.status_bar.config(text=label)

    def _on_done(self, result):
        self.run_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.open_btn.config(state=tk.NORMAL)
        self.progress_var.set(100)
        self.progress_label.config(text="训练完成！")
        self.status_bar.config(text=f"完成 - 输出目录: {result.get('output_dir','')}")

        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, "=" * 50 + "\n")
        self.result_text.insert(tk.END, "   AML 反洗钱检测结果\n")
        self.result_text.insert(tk.END, "=" * 50 + "\n\n")
        self.result_text.insert(tk.END, f"AUC-ROC:     {result['auc']:.4f}\n")
        self.result_text.insert(tk.END, f"召回率:      {result['recall']:.4f}\n")
        self.result_text.insert(tk.END, f"精确率:      {result['precision']:.4f}\n")
        self.result_text.insert(tk.END, f"最佳阈值:    {result['best_th']:.2f}\n")
        self.result_text.insert(tk.END, f"测试样本:    {result['total_test']}\n")
        self.result_text.insert(tk.END, f"识别可疑:    {result['suspicious_count']} 个\n")
        self.result_text.insert(tk.END, f"其中真实:    {result['suspicious_true']} 个\n")
        self.result_text.insert(tk.END, f"\n输出目录:    {result['output_dir']}\n")
        self.result_text.insert(tk.END, "\n--- 分类报告 ---\n")
        self.result_text.insert(tk.END, result['report'])

        messagebox.showinfo("完成", f"训练完成！\n\nAUC: {result['auc']:.4f}\n识别可疑账户: {result['suspicious_count']} 个\n\n结果已保存到 output 目录")

    def _on_error(self, msg):
        self.run_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_label.config(text="训练出错！")
        self.status_bar.config(text="错误")
        self._log(f"发生错误: {msg}")
        messagebox.showerror("错误", f"训练过程中发生错误:\n\n{msg}")

    def _open_output(self):
        output_dir = os.path.join(APP_DIR, 'output')
        if os.path.exists(output_dir):
            os.startfile(output_dir)
        else:
            messagebox.showinfo("提示", "输出目录尚未生成，请先完成训练。")

def main():
    root = tk.Tk()
    app = AMLApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
