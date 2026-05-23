"""
AML 反洗钱检测系统 v11 - Gradio 可视化界面
内核：aml_model_v11.py（混合采样 + 集成学习 + F1.5阈值优化）
启动：python aml_ui_v11.py
"""

import gradio as gr
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import io
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_score, recall_score, fbeta_score)
from imblearn.combine import SMOTETomek

# ---------- 字体 ----------
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

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

# ==================== 全局状态 ====================
STATE = {}

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
    sg = trans.groupby('SENDER_ACCOUNT_ID')
    s_ad = sg['day'].nunique().to_dict()
    s_dt = (sg.size() / sg['day'].nunique()).to_dict()
    s_na = trans[trans['is_night']].groupby('SENDER_ACCOUNT_ID')['TX_AMOUNT'].sum().to_dict()
    s_ta = sg['TX_AMOUNT'].sum().to_dict()
    s_nr = {a: s_na.get(a,0)/(s_ta.get(a,0)+1e-8) for a in set(s_na)|set(s_ta)}
    s_dam = sg.apply(lambda x: x.groupby('day')['TX_AMOUNT'].sum().max()).to_dict()
    s_dcm = sg.apply(lambda x: x.groupby('day').size().max()).to_dict()
    rg = trans.groupby('RECEIVER_ACCOUNT_ID')
    r_ad = rg['day'].nunique().to_dict()
    r_dt = (rg.size() / rg['day'].nunique()).to_dict()
    r_na = trans[trans['is_night']].groupby('RECEIVER_ACCOUNT_ID')['TX_AMOUNT'].sum().to_dict()
    r_ta = rg['TX_AMOUNT'].sum().to_dict()
    r_nr = {a: r_na.get(a,0)/(r_ta.get(a,0)+1e-8) for a in set(r_na)|set(r_ta)}
    r_dam = rg.apply(lambda x: x.groupby('day')['TX_AMOUNT'].sum().max()).to_dict()
    r_dcm = rg.apply(lambda x: x.groupby('day').size().max()).to_dict()
    all_acc = accounts['ACCOUNT_ID'].unique()
    tf = pd.DataFrame({'ACCOUNT_ID': all_acc})
    for col, d in [('active_days_send', s_ad), ('daily_tx_send', s_dt), ('night_ratio_send', s_nr),
                   ('max_daily_amount_send', s_dam), ('max_daily_count_send', s_dcm),
                   ('active_days_recv', r_ad), ('daily_tx_recv', r_dt), ('night_ratio_recv', r_nr),
                   ('max_daily_amount_recv', r_dam), ('max_daily_count_recv', r_dcm)]:
        tf[col] = tf['ACCOUNT_ID'].map(d).fillna(0)
    return tf

def build_graph_and_extract_network_features(trans, accounts):
    G = nx.from_pandas_edgelist(trans, source='SENDER_ACCOUNT_ID',
                                target='RECEIVER_ACCOUNT_ID', create_using=nx.DiGraph())
    nodes = list(accounts['ACCOUNT_ID'].values)
    nf = pd.DataFrame({'ACCOUNT_ID': nodes})
    nf['out_degree'] = nf['ACCOUNT_ID'].map(dict(G.out_degree(nodes))).fillna(0)
    nf['in_degree'] = nf['ACCOUNT_ID'].map(dict(G.in_degree(nodes))).fillna(0)
    nf['total_degree'] = nf['out_degree'] + nf['in_degree']
    nf['pagerank'] = nf['ACCOUNT_ID'].map(nx.pagerank(G, alpha=0.85, max_iter=100)).fillna(0)
    nf['clustering'] = nf['ACCOUNT_ID'].map(nx.clustering(G.to_undirected())).fillna(0)
    return nf

def extract_features(trans, accounts):
    # 统计特征
    ss = trans.groupby('SENDER_ACCOUNT_ID').agg(
        send_sum=('TX_AMOUNT','sum'), send_mean=('TX_AMOUNT','mean'),
        send_max=('TX_AMOUNT','max'), send_std=('TX_AMOUNT','std'),
        send_count=('TX_AMOUNT','count')).reset_index().rename(columns={'SENDER_ACCOUNT_ID':'ACCOUNT_ID'})
    rs = trans.groupby('RECEIVER_ACCOUNT_ID').agg(
        recv_sum=('TX_AMOUNT','sum'), recv_mean=('TX_AMOUNT','mean'),
        recv_max=('TX_AMOUNT','max'), recv_std=('TX_AMOUNT','std'),
        recv_count=('TX_AMOUNT','count')).reset_index().rename(columns={'RECEIVER_ACCOUNT_ID':'ACCOUNT_ID'})
    af = accounts[['ACCOUNT_ID','IS_FRAUD','INIT_BALANCE','COUNTRY','ACCOUNT_TYPE','TX_BEHAVIOR_ID']].copy()
    af = af.merge(ss, on='ACCOUNT_ID', how='left')
    af = af.merge(rs, on='ACCOUNT_ID', how='left')
    fill_cols = ['send_sum','send_mean','send_max','send_std','send_count',
                 'recv_sum','recv_mean','recv_max','recv_std','recv_count']
    af[fill_cols] = af[fill_cols].fillna(0)
    af['out_in_ratio'] = af['send_count'] / (af['recv_count'] + 1e-6)
    af['net_flow'] = af['send_sum'] - af['recv_sum']
    # 网络特征
    nf = build_graph_and_extract_network_features(trans, accounts)
    af = af.merge(nf, on='ACCOUNT_ID', how='left')
    # 时序特征
    tf = extract_temporal_features(trans, accounts)
    af = af.merge(tf, on='ACCOUNT_ID', how='left')
    # 类别编码（频率编码）
    if 'COUNTRY' in af.columns:
        freq = accounts['COUNTRY'].value_counts(normalize=True).to_dict()
        af['country_freq'] = af['COUNTRY'].map(freq).fillna(0)
        af.drop('COUNTRY', axis=1, inplace=True)
    if 'ACCOUNT_TYPE' in af.columns:
        dum = pd.get_dummies(af['ACCOUNT_TYPE'], prefix='type')
        af = pd.concat([af, dum], axis=1)
        af.drop('ACCOUNT_TYPE', axis=1, inplace=True)
    if 'TX_BEHAVIOR_ID' in af.columns:
        freq = accounts['TX_BEHAVIOR_ID'].value_counts(normalize=True).to_dict()
        af['behavior_freq'] = af['TX_BEHAVIOR_ID'].map(freq).fillna(0)
        af.drop('TX_BEHAVIOR_ID', axis=1, inplace=True)

    exclude = ['ACCOUNT_ID', 'IS_FRAUD']
    feats = [c for c in af.columns if c not in exclude]
    X = af[feats].fillna(0)
    y = af['IS_FRAUD']
    return X, y, feats, af

def find_threshold_fbeta(y_val, y_val_proba, beta=1.5):
    thresholds = np.linspace(0.01, 0.99, 200)
    best_th, best_fb = 0.5, 0
    for th in thresholds:
        fb = fbeta_score(y_val, (y_val_proba >= th).astype(int), beta=beta)
        if fb > best_fb:
            best_fb, best_th = fb, th
    return best_th

def build_test_features(trans_train_val, trans_test, accounts, feature_cols):
    """构建测试集特征（复用训练图）"""
    ss = trans_test.groupby('SENDER_ACCOUNT_ID').agg(
        send_sum=('TX_AMOUNT','sum'), send_mean=('TX_AMOUNT','mean'),
        send_max=('TX_AMOUNT','max'), send_std=('TX_AMOUNT','std'),
        send_count=('TX_AMOUNT','count')).reset_index().rename(columns={'SENDER_ACCOUNT_ID':'ACCOUNT_ID'})
    rs = trans_test.groupby('RECEIVER_ACCOUNT_ID').agg(
        recv_sum=('TX_AMOUNT','sum'), recv_mean=('TX_AMOUNT','mean'),
        recv_max=('TX_AMOUNT','max'), recv_std=('TX_AMOUNT','std'),
        recv_count=('TX_AMOUNT','count')).reset_index().rename(columns={'RECEIVER_ACCOUNT_ID':'ACCOUNT_ID'})
    at = accounts.copy()
    at = at.merge(ss, on='ACCOUNT_ID', how='left')
    at = at.merge(rs, on='ACCOUNT_ID', how='left')
    fill_cols = ['send_sum','send_mean','send_max','send_std','send_count',
                 'recv_sum','recv_mean','recv_max','recv_std','recv_count']
    at[fill_cols] = at[fill_cols].fillna(0)
    at['out_in_ratio'] = at['send_count'] / (at['recv_count'] + 1e-6)
    at['net_flow'] = at['send_sum'] - at['recv_sum']
    # 网络特征（复用训练图）
    G = nx.from_pandas_edgelist(trans_train_val, source='SENDER_ACCOUNT_ID',
                                target='RECEIVER_ACCOUNT_ID', create_using=nx.DiGraph())
    nodes = list(at['ACCOUNT_ID'].values)
    nf = pd.DataFrame({'ACCOUNT_ID': nodes})
    nf['out_degree'] = nf['ACCOUNT_ID'].map(dict(G.out_degree(nodes))).fillna(0)
    nf['in_degree'] = nf['ACCOUNT_ID'].map(dict(G.in_degree(nodes))).fillna(0)
    nf['total_degree'] = nf['out_degree'] + nf['in_degree']
    nf['pagerank'] = nf['ACCOUNT_ID'].map(nx.pagerank(G, alpha=0.85, max_iter=100)).fillna(0)
    nf['clustering'] = nf['ACCOUNT_ID'].map(nx.clustering(G.to_undirected())).fillna(0)
    at = at.merge(nf, on='ACCOUNT_ID', how='left')
    # 时序
    tf = extract_temporal_features(trans_test, accounts)
    at = at.merge(tf, on='ACCOUNT_ID', how='left')
    # 类别编码
    if 'COUNTRY' in at.columns:
        at['country_freq'] = at['COUNTRY'].map(at['COUNTRY'].value_counts(normalize=True).to_dict()).fillna(0)
        at.drop('COUNTRY', axis=1, inplace=True)
    if 'ACCOUNT_TYPE' in at.columns:
        dum = pd.get_dummies(at['ACCOUNT_TYPE'], prefix='type')
        at = pd.concat([at, dum], axis=1)
        at.drop('ACCOUNT_TYPE', axis=1, inplace=True)
    if 'TX_BEHAVIOR_ID' in at.columns:
        at['behavior_freq'] = at['TX_BEHAVIOR_ID'].map(at['TX_BEHAVIOR_ID'].value_counts(normalize=True).to_dict()).fillna(0)
        at.drop('TX_BEHAVIOR_ID', axis=1, inplace=True)
    X_test = at[feature_cols].fillna(0)
    y_test = at['IS_FRAUD']
    return X_test, y_test, at

# ==================== Gradio 回调 ====================

def do_load_data(trans_file, acc_file):
    """上传并加载数据"""
    if trans_file is None or acc_file is None:
        return "请上传两个数据文件", "", ""
    try:
        trans = pd.read_csv(trans_file.name)
        accounts = pd.read_csv(acc_file.name)
        STATE['trans'] = trans
        STATE['accounts'] = accounts
        info = f"""
### ✅ 数据加载成功

| 指标 | 数值 |
|------|------|
| 交易笔数 | {len(trans):,} |
| 账户数量 | {len(accounts):,} |
| 可疑账户 | {accounts['IS_FRAUD'].sum():,} ({accounts['IS_FRAUD'].mean():.2%}) |
| 交易时间范围 | {trans['TIMESTAMP'].min()} ~ {trans['TIMESTAMP'].max()} |
        """
        return info, ", ".join(trans.columns.tolist()), ", ".join(accounts.columns.tolist())
    except Exception as e:
        return f"❌ 加载失败: {str(e)}", "", ""

def do_sample(data_type, n_rows):
    """浏览数据样本"""
    data = STATE.get('trans' if data_type == "交易数据" else 'accounts')
    if data is None:
        return "请先加载数据"
    return data.head(n_rows)

def do_plot_distribution():
    """数据分布可视化"""
    trans = STATE.get('trans')
    accounts = STATE.get('accounts')
    if trans is None or accounts is None:
        return None, "请先加载数据"
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # 1 交易金额
    s = trans.sample(min(10000, len(trans)))
    axes[0,0].hist(s['TX_AMOUNT'], bins=50, edgecolor='black', alpha=0.7)
    axes[0,0].set_title('交易金额分布（采样）')
    # 2 账户类型
    tc = accounts['ACCOUNT_TYPE'].value_counts()
    axes[0,1].pie(tc.values, labels=tc.index, autopct='%1.1f%%')
    axes[0,1].set_title('账户类型分布')
    # 3 可疑分布
    fc = accounts['IS_FRAUD'].value_counts()
    axes[1,0].bar(['正常','可疑'], fc.values, color=['#3498db','#e74c3c'])
    axes[1,0].set_title('正常 vs 可疑')
    # 4 交易类型
    ttc = trans['TX_TYPE'].value_counts().head(10)
    axes[1,1].barh(ttc.index, ttc.values, color='skyblue')
    axes[1,1].set_title('交易类型 Top 10')
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0); img = Image.open(buf); plt.close()
    return img, "图表已生成"

def do_train(rf_weight, xgb_weight, progress=gr.Progress()):
    """v11 完整训练流程"""
    trans = STATE.get('trans')
    accounts = STATE.get('accounts')
    if trans is None or accounts is None:
        return None, None, None, "请先加载数据", gr.update(visible=False)

    total = rf_weight + xgb_weight
    rf_w = rf_weight / total if total > 0 else 0.5
    xgb_w = xgb_weight / total if total > 0 else 0.5
    STATE['rf_w'] = rf_w
    STATE['xgb_w'] = xgb_w

    try:
        progress(0.05, desc="时间划分...")
        trans_train, trans_val, trans_test = split_by_time(trans)
        trans_train_val = pd.concat([trans_train, trans_val])
        log = f"""
### ▶ 开始训练

| 阶段 | 详情 |
|------|------|
| 训练交易 | {len(trans_train):,} 笔 |
| 验证交易 | {len(trans_val):,} 笔 |
| 测试交易 | {len(trans_test):,} 笔 |
| 可疑占比 | {accounts['IS_FRAUD'].mean():.2%} |
| RF权重 | {rf_w:.2f} |
| XGB权重 | {xgb_w:.2f} |
"""

        progress(0.15, desc="提取特征...")
        X_tv, y_tv, feature_cols, _ = extract_features(trans_train_val, accounts)
        STATE['feature_cols'] = feature_cols
        log += f"\n特征维度: {X_tv.shape}"

        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=0.2, random_state=42, stratify=y_tv)

        progress(0.35, desc="SMOTETomek 混合采样...")
        smt = SMOTETomek(random_state=42)
        X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
        log += f"\n采样后样本: {len(X_train_res)} (可疑: {y_train_res.sum()})"

        progress(0.50, desc="训练随机森林...")
        if USE_GPU_RF:
            rf = CumlRF(random_state=42, n_estimators=200, max_depth=None,
                        min_samples_split=2, class_weight='balanced_subsample')
            rf.fit(X_train_res.values, y_train_res.values)
        else:
            rf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=None,
                                        min_samples_split=2, class_weight='balanced_subsample', n_jobs=-1)
            rf.fit(X_train_res, y_train_res)

        if XGB_AVAILABLE:
            progress(0.65, desc="训练 XGBoost...")
            xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                       random_state=42, eval_metric='logloss')
            xgb_m.fit(X_train_res, y_train_res)
        else:
            xgb_m = None

        progress(0.75, desc="F1.5 阈值优化...")
        rf_val_p = rf.predict_proba(X_val)[:, 1]
        if xgb_m:
            y_val_p = rf_w * rf_val_p + xgb_w * xgb_m.predict_proba(X_val)[:, 1]
        else:
            y_val_p = rf_val_p
        best_th = find_threshold_fbeta(y_val, y_val_p)

        progress(0.85, desc="构建测试集特征...")
        X_test, y_test, account_test = build_test_features(trans_train_val, trans_test, accounts, feature_cols)
        rf_test_p = rf.predict_proba(X_test)[:, 1]
        if xgb_m:
            y_test_p = rf_w * rf_test_p + xgb_w * xgb_m.predict_proba(X_test)[:, 1]
        else:
            y_test_p = rf_test_p
        y_test_pred = (y_test_p >= best_th).astype(int)

        progress(0.95, desc="生成结果...")
        STATE['rf'] = rf; STATE['xgb_m'] = xgb_m
        STATE['best_th'] = best_th; STATE['X_test'] = X_test
        STATE['y_test'] = y_test; STATE['y_test_p'] = y_test_p
        STATE['y_test_pred'] = y_test_pred; STATE['account_test'] = account_test

        # 评估结果
        auc = roc_auc_score(y_test, y_test_p)
        rec = recall_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred)
        report = classification_report(y_test, y_test_pred, target_names=['正常','可疑'])
        n_susp = y_test_pred.sum()
        n_true_susp = int((y_test_pred & y_test.astype(bool)).sum())

        result = f"""
### ✅ 训练完成

| 指标 | 数值 |
|------|------|
| AUC-ROC | **{auc:.4f}** |
| 召回率(Recall) | **{rec:.4f}** |
| 精确率(Precision) | **{prec:.4f}** |
| 最佳阈值 | {best_th:.2f} |
| 测试样本 | {len(y_test)} |
| 识别可疑账户 | {n_susp} 个 |
| 其中真实可疑 | {n_true_susp} 个 |
"""

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_test_pred)
        fig1, ax1 = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['正常','可疑'], yticklabels=['正常','可疑'], ax=ax1)
        ax1.set_xlabel('预测'); ax1.set_ylabel('真实'); ax1.set_title(f'混淆矩阵 (阈值={best_th:.2f})')
        buf1 = io.BytesIO(); plt.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
        buf1.seek(0); cm_img = Image.open(buf1); plt.close()

        # 特征重要性
        if not USE_GPU_RF:
            imps = rf.feature_importances_
            idxs = np.argsort(imps)[::-1][:20]
            fig2, ax2 = plt.subplots(figsize=(10,7))
            ax2.barh(range(len(idxs)), imps[idxs], color='steelblue')
            ax2.set_yticks(range(len(idxs)))
            ax2.set_yticklabels([feature_cols[i] for i in idxs])
            ax2.invert_yaxis(); ax2.set_title('特征重要性 Top 20')
            plt.tight_layout()
            buf2 = io.BytesIO(); plt.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
            buf2.seek(0); fi_img = Image.open(buf2); plt.close()
        else:
            fi_img = None

        # SHAP 汇总图
        if not USE_GPU_RF:
            try:
                X_samp = X_test.sample(min(150, len(X_test)), random_state=42)
                explainer = shap.TreeExplainer(rf)
                sv = explainer.shap_values(X_samp)
                sv_c = sv[1] if isinstance(sv, list) else sv[:,:,1]
                shap.summary_plot(sv_c, X_samp, feature_names=feature_cols, show=False)
                buf3 = io.BytesIO(); plt.savefig(buf3, format='png', dpi=100, bbox_inches='tight')
                buf3.seek(0); shap_img = Image.open(buf3); plt.close()
            except:
                shap_img = None
        else:
            shap_img = None

        progress(1.0, desc="完成！")
        return cm_img, fi_img, shap_img, log + "\n--- 分类报告 ---\n" + report, gr.update(visible=True)

    except Exception as e:
        import traceback
        return None, None, None, f"❌ 训练失败: {str(e)}\n{traceback.format_exc()}", gr.update(visible=False)

def do_predict(account_id):
    """单账户预测"""
    if 'rf' not in STATE:
        return "请先训练模型", None, None
    try:
        aid = int(account_id)
        row = STATE['account_test']
        row = row[row['ACCOUNT_ID'] == aid]
        if len(row) == 0:
            return f"未找到账户 {aid}", None, None
        idx = row.index[0]
        proba = STATE['y_test_p'][idx]
        pred = int(proba >= STATE['best_th'])
        result = f"""
### 账户 {aid} 预测

| 项目 | 数值 |
|------|------|
| 预测类别 | **{'🔴 可疑' if pred else '🟢 正常'}** |
| 可疑概率 | {proba:.4f} ({proba*100:.2f}%) |
| 真实标签 | {'可疑' if STATE['y_test'].iloc[idx] else '正常'} |
| 初始余额 | {row['INIT_BALANCE'].values[0]:,.2f} |
| 发送总额 | {row['send_sum'].values[0]:,.2f} |
| 接收总额 | {row['recv_sum'].values[0]:,.2f} |
        """
        return result, gr.update(visible=True), row[['ACCOUNT_ID','send_sum','recv_sum','send_count','recv_count']].head(1)
    except Exception as e:
        return f"预测失败: {e}", None, None

def do_explain(account_id):
    """SHAP瀑布图"""
    if 'rf' not in STATE:
        return None, "请先训练模型"
    try:
        aid = int(account_id)
        row = STATE['account_test']
        row = row[row['ACCOUNT_ID'] == aid]
        if len(row) == 0:
            return None, f"未找到账户 {aid}"
        idx = row.index[0]
        fc = STATE['feature_cols']
        X_row = STATE['X_test'].iloc[idx:idx+1][fc].fillna(0)
        rf = STATE['rf']
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(X_row)
        sv_c = sv[1][0] if isinstance(sv, list) else sv[0]
        ev = explainer.expected_value
        bv = float(ev[1]) if isinstance(ev,(list,np.ndarray)) and len(ev)>=2 else float(ev)
        exp = shap.Explanation(values=sv_c, base_values=bv, data=X_row.iloc[0].values, feature_names=fc)
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.waterfall_plot(exp, show=False)
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0); img = Image.open(buf); plt.close()
        return img, f"账户 {aid} SHAP 解释"
    except Exception as e:
        return None, f"SHAP失败: {e}"

def do_batch(top_n):
    """批量预测"""
    if 'rf' not in STATE:
        return None, "请先训练模型"
    try:
        df = STATE['account_test'].copy()
        df['proba'] = STATE['y_test_p']
        df['pred'] = (df['proba'] >= STATE['best_th']).astype(int)
        top = df[df['pred']==1].nlargest(top_n, 'proba')[
            ['ACCOUNT_ID','IS_FRAUD','proba','INIT_BALANCE','send_sum','recv_sum','send_count','recv_count']]
        top['proba'] = top['proba'].apply(lambda x: f"{x:.4f}")
        for c in ['INIT_BALANCE','send_sum','recv_sum']:
            top[c] = top[c].apply(lambda x: f"{x:,.2f}")
        return top, f"Top {top_n} 可疑账户 ({len(top)} 个)"
    except Exception as e:
        return None, f"批量预测失败: {e}"

# ==================== Gradio 界面 ====================

def create_ui():
    with gr.Blocks(title="AML 反洗钱检测系统 v11", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏦 AML 反洗钱检测系统 v11
        **内核**: 混合采样(SMOTETomek) + 集成学习(RF+XGBoost) + F1.5阈值优化
        """)

        with gr.Tab("📁 数据导入"):
            with gr.Row():
                trans_file = gr.File(label="交易数据 (transactions.csv)", file_types=[".csv"])
                acc_file = gr.File(label="账户数据 (accounts.csv)", file_types=[".csv"])
            load_btn = gr.Button("加载数据", variant="primary")
            data_info = gr.Markdown()
            with gr.Row():
                trans_cols = gr.Textbox(label="交易列名", lines=2, interactive=False)
                acc_cols = gr.Textbox(label="账户列名", lines=2, interactive=False)

        with gr.Tab("🔍 数据探索"):
            with gr.Row():
                data_type = gr.Radio(["交易数据","账户数据"], value="交易数据", label="类型")
                n_rows = gr.Slider(5, 50, 10, 5, label="行数")
                sample_btn = gr.Button("查看样本")
            sample_out = gr.DataFrame()
            viz_btn = gr.Button("生成分布图", variant="primary")
            viz_img = gr.Image()
            viz_status = gr.Textbox(interactive=False)

        with gr.Tab("🤖 模型训练"):
            with gr.Row():
                rf_w = gr.Slider(0, 1, 0.5, 0.05, label="RF 权重")
                xgb_w = gr.Slider(0, 1, 0.5, 0.05, label="XGB 权重")
            train_btn = gr.Button("▶ 开始训练", variant="primary", size="lg")
            train_log = gr.Markdown()
            with gr.Row():
                cm_img = gr.Image(label="混淆矩阵")
                fi_img = gr.Image(label="特征重要性")
            shap_img = gr.Image(label="SHAP 汇总图")

        with gr.Tab("🎯 预测分析") as pred_tab:
            with gr.Row():
                aid_input = gr.Number(label="账户ID", precision=0)
                pred_btn = gr.Button("预测", variant="primary")
            pred_result = gr.Markdown()
            with gr.Row():
                explain_btn = gr.Button("SHAP 瀑布图", variant="secondary", visible=False)
                explain_img = gr.Image(label="SHAP 解释")
                explain_status = gr.Textbox(interactive=False)
            pred_detail = gr.DataFrame(label="账户详情")
            gr.Markdown("---\n### 批量预测")
            top_n = gr.Slider(5, 50, 20, 5, label="前N个可疑账户")
            batch_btn = gr.Button("批量预测", variant="primary")
            batch_out = gr.DataFrame()
            batch_status = gr.Textbox(interactive=False)

        # 事件绑定
        load_btn.click(do_load_data, [trans_file, acc_file], [data_info, trans_cols, acc_cols])
        sample_btn.click(do_sample, [data_type, n_rows], [sample_out])
        viz_btn.click(do_plot_distribution, [], [viz_img, viz_status])
        train_btn.click(do_train, [rf_w, xgb_w], [cm_img, fi_img, shap_img, train_log, pred_tab])
        pred_btn.click(do_predict, [aid_input], [pred_result, explain_btn, pred_detail])
        explain_btn.click(do_explain, [aid_input], [explain_img, explain_status])
        batch_btn.click(do_batch, [top_n], [batch_out, batch_status])

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
