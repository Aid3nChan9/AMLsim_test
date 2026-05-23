"""
AML 反洗钱检测系统 - Gradio 可视化界面
功能：数据导入、探索可视化、模型训练、预测解释
"""

import gradio as gr
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import io
import base64
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 全局变量 =====================
global_data = {
    'transactions': None,
    'accounts': None,
    'features': None,
    'model': None,
    'feature_cols': None,
    'X_test': None,
    'y_test': None,
    'explainer': None
}

# ===================== 核心功能函数 =====================

def build_features(trans, accounts):
    """构建特征工程"""
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
    
    # 合并基础信息
    account_features = accounts[['ACCOUNT_ID', 'IS_FRAUD', 'INIT_BALANCE', 
                                  'COUNTRY', 'ACCOUNT_TYPE', 'TX_BEHAVIOR_ID']].copy()
    account_features = account_features.merge(sender_stats, on='ACCOUNT_ID', how='left')
    account_features = account_features.merge(receiver_stats, on='ACCOUNT_ID', how='left')
    
    # 填充缺失值
    fill_cols = ['send_sum', 'send_mean', 'send_max', 'send_std', 'send_count',
                 'recv_sum', 'recv_mean', 'recv_max', 'recv_std', 'recv_count']
    account_features[fill_cols] = account_features[fill_cols].fillna(0)
    
    # 衍生特征
    account_features['out_in_ratio'] = account_features['send_count'] / (account_features['recv_count'] + 1e-6)
    account_features['net_flow'] = account_features['send_sum'] - account_features['recv_sum']
    
    return account_features

def build_network_features(trans, accounts, progress=None):
    """构建网络拓扑特征"""
    G = nx.DiGraph()
    edges = list(zip(trans['SENDER_ACCOUNT_ID'], trans['RECEIVER_ACCOUNT_ID']))
    G.add_edges_from(edges)
    
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

def encode_features(account_features):
    """编码类别特征"""
    le_country = LabelEncoder()
    le_type = LabelEncoder()
    le_behavior = LabelEncoder()
    
    account_features['country_enc'] = le_country.fit_transform(account_features['COUNTRY'].astype(str))
    account_features['type_enc'] = le_type.fit_transform(account_features['ACCOUNT_TYPE'].astype(str))
    account_features['behavior_enc'] = le_behavior.fit_transform(
        account_features['TX_BEHAVIOR_ID'].fillna(-1).astype(int))
    
    return account_features

# ===================== UI 回调函数 =====================

def load_data(trans_file, acc_file):
    """加载数据文件"""
    try:
        if trans_file is None or acc_file is None:
            return "请上传两个数据文件", None, None, None
        
        # 读取文件
        trans = pd.read_csv(trans_file.name)
        accounts = pd.read_csv(acc_file.name)
        
        global_data['transactions'] = trans
        global_data['accounts'] = accounts
        
        # 基础统计
        trans_count = len(trans)
        acc_count = len(accounts)
        fraud_ratio = accounts['IS_FRAUD'].mean()
        
        info = f"""
### 数据加载成功！

| 指标 | 数值 |
|------|------|
| 交易笔数 | {trans_count:,} |
| 账户数量 | {acc_count:,} |
| 可疑账户占比 | {fraud_ratio:.2%} |
| 交易时间范围 | {trans['TIMESTAMP'].min()} ~ {trans['TIMESTAMP'].max()} |
        """
        
        # 显示列名
        trans_cols = ", ".join(trans.columns.tolist())
        acc_cols = ", ".join(accounts.columns.tolist())
        
        return info, trans_cols, acc_cols, gr.update(visible=True)
    except Exception as e:
        return f"数据加载失败: {str(e)}", None, None, gr.update(visible=False)

def show_data_sample(data_type, n_rows):
    """显示数据样本"""
    if data_type == "交易数据":
        data = global_data['transactions']
    else:
        data = global_data['accounts']
    
    if data is None:
        return "请先加载数据"
    
    return data.head(n_rows)

def plot_data_distribution():
    """绘制数据分布图"""
    trans = global_data['transactions']
    accounts = global_data['accounts']
    
    if trans is None or accounts is None:
        return None, "请先加载数据"
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 交易金额分布
    ax1 = axes[0, 0]
    sample_trans = trans.sample(min(10000, len(trans)))
    ax1.hist(sample_trans['TX_AMOUNT'], bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('交易金额')
    ax1.set_ylabel('频次')
    ax1.set_title('交易金额分布（采样）')
    
    # 2. 账户类型分布
    ax2 = axes[0, 1]
    type_counts = accounts['ACCOUNT_TYPE'].value_counts()
    ax2.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
    ax2.set_title('账户类型分布')
    
    # 3. 可疑账户分布
    ax3 = axes[1, 0]
    fraud_counts = accounts['IS_FRAUD'].value_counts()
    colors = ['#3498db', '#e74c3c']
    ax3.bar(['正常', '可疑'], fraud_counts.values, color=colors, edgecolor='black')
    ax3.set_ylabel('账户数量')
    ax3.set_title('正常 vs 可疑账户分布')
    
    # 4. 交易类型分布
    ax4 = axes[1, 1]
    tx_type_counts = trans['TX_TYPE'].value_counts().head(10)
    ax4.barh(tx_type_counts.index, tx_type_counts.values, color='skyblue', edgecolor='black')
    ax4.set_xlabel('交易笔数')
    ax4.set_title('交易类型分布（Top 10）')
    
    plt.tight_layout()
    
    # 转换为图像
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img, "数据分布可视化完成"

def train_model(n_estimators, max_depth, min_samples_split, test_size, use_network):
    """训练模型"""
    trans = global_data['transactions']
    accounts = global_data['accounts']
    
    if trans is None or accounts is None:
        return None, None, None, "请先加载数据"
    
    try:
        # 构建基础特征
        account_features = build_features(trans, accounts)
        
        # 构建网络特征（可选）
        if use_network:
            network_feat = build_network_features(trans, accounts)
            account_features = account_features.merge(network_feat, on='ACCOUNT_ID', how='left')
            network_cols = ['out_degree', 'in_degree', 'total_degree', 'pagerank', 'clustering']
        else:
            network_cols = []
        
        # 编码
        account_features = encode_features(account_features)
        
        # 特征列
        feature_cols = [
            'INIT_BALANCE', 'send_sum', 'send_mean', 'send_max', 'send_std', 'send_count',
            'recv_sum', 'recv_mean', 'recv_max', 'recv_std', 'recv_count',
            'out_in_ratio', 'net_flow',
            'country_enc', 'type_enc', 'behavior_enc'
        ] + network_cols
        
        global_data['feature_cols'] = feature_cols
        
        X = account_features[feature_cols]
        y = account_features['IS_FRAUD']
        
        # 处理缺失值
        X = X.fillna(0)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
        
        global_data['X_test'] = X_test
        global_data['y_test'] = y_test
        
        # 训练模型
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf.fit(X_train, y_train)
        
        global_data['model'] = rf
        global_data['features'] = account_features
        
        # 预测
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]
        
        # 评估指标
        auc = roc_auc_score(y_test, y_proba)
        
        # 分类报告
        report = classification_report(y_test, y_pred, target_names=['正常', '可疑'], output_dict=True)
        
        result_text = f"""
### 模型训练完成！

| 指标 | 数值 |
|------|------|
| AUC-ROC | {auc:.4f} |
| 正常账户精确率 | {report['正常']['precision']:.4f} |
| 正常账户召回率 | {report['正常']['recall']:.4f} |
| 正常账户F1 | {report['正常']['f1-score']:.4f} |
| 可疑账户精确率 | {report['可疑']['precision']:.4f} |
| 可疑账户召回率 | {report['可疑']['recall']:.4f} |
| 可疑账户F1 | {report['可疑']['f1-score']:.4f} |

特征数量: {len(feature_cols)}
训练样本: {len(X_train)}
测试样本: {len(X_test)}
        """
        
        # 混淆矩阵图
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['正常','可疑'], yticklabels=['正常','可疑'], ax=ax)
        ax.set_xlabel('预测')
        ax.set_ylabel('真实')
        ax.set_title('混淆矩阵')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        cm_img = Image.open(buf)
        plt.close()
        
        # 特征重要性图
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(indices)), importances[indices], align='center', color='steelblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_cols[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('重要性')
        ax.set_title('特征重要性 (Top 15)')
        
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
        buf2.seek(0)
        fi_img = Image.open(buf2)
        plt.close()
        
        return cm_img, fi_img, result_text, gr.update(visible=True)
        
    except Exception as e:
        return None, None, f"训练失败: {str(e)}", gr.update(visible=False)

def predict_account(account_id):
    """预测单个账户"""
    model = global_data['model']
    features = global_data['features']
    feature_cols = global_data['feature_cols']
    
    if model is None:
        return "请先训练模型", None
    
    try:
        account_id = int(account_id)
        account_data = features[features['ACCOUNT_ID'] == account_id]
        
        if len(account_data) == 0:
            return f"未找到账户ID: {account_id}", None
        
        X = account_data[feature_cols].fillna(0)
        proba = model.predict_proba(X)[0, 1]
        pred = model.predict(X)[0]
        
        result = f"""
### 预测结果

| 项目 | 数值 |
|------|------|
| 账户ID | {account_id} |
| 预测类别 | {'可疑' if pred == 1 else '正常'} |
| 可疑概率 | {proba:.4f} ({proba*100:.2f}%) |
| 初始余额 | {account_data['INIT_BALANCE'].values[0]:,.2f} |
| 发送总额 | {account_data['send_sum'].values[0]:,.2f} |
| 接收总额 | {account_data['recv_sum'].values[0]:,.2f} |
| 交易次数 | 发送 {account_data['send_count'].values[0]:.0f} / 接收 {account_data['recv_count'].values[0]:.0f} |
        """
        
        return result, gr.update(visible=True)
        
    except Exception as e:
        return f"预测失败: {str(e)}", None

def explain_prediction(account_id):
    """SHAP解释预测"""
    model = global_data['model']
    features = global_data['features']
    feature_cols = global_data['feature_cols']
    
    if model is None:
        return None, "请先训练模型"
    
    try:
        account_id = int(account_id)
        account_data = features[features['ACCOUNT_ID'] == account_id]
        
        if len(account_data) == 0:
            return None, f"未找到账户ID: {account_id}"
        
        X = account_data[feature_cols].fillna(0)
        
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # 处理二分类
        if isinstance(shap_values, list):
            shap_values_class = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values_class = shap_values[:, :, 1]
        else:
            shap_values_class = shap_values
        
        # 瀑布图
        exp_val = explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            base_val = float(exp_val[1]) if len(exp_val) >= 2 else float(exp_val[0])
        else:
            base_val = float(exp_val)
        
        exp = shap.Explanation(
            values=shap_values_class[0],
            base_values=base_val,
            data=X.iloc[0, :].values,
            feature_names=feature_cols
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.waterfall_plot(exp, show=False)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return img, "SHAP解释生成完成"
        
    except Exception as e:
        return None, f"解释失败: {str(e)}"

def batch_predict(top_n):
    """批量预测最可疑的账户"""
    model = global_data['model']
    features = global_data['features']
    feature_cols = global_data['feature_cols']
    
    if model is None:
        return None, "请先训练模型"
    
    try:
        X = features[feature_cols].fillna(0)
        proba = model.predict_proba(X)[:, 1]
        
        features_copy = features.copy()
        features_copy['fraud_proba'] = proba
        
        # 获取最可疑的账户
        top_suspicious = features_copy.nlargest(top_n, 'fraud_proba')[
            ['ACCOUNT_ID', 'IS_FRAUD', 'fraud_proba', 'INIT_BALANCE', 
             'send_sum', 'recv_sum', 'send_count', 'recv_count']
        ]
        
        top_suspicious['fraud_proba'] = top_suspicious['fraud_proba'].apply(lambda x: f"{x:.4f}")
        top_suspicious['INIT_BALANCE'] = top_suspicious['INIT_BALANCE'].apply(lambda x: f"{x:,.2f}")
        top_suspicious['send_sum'] = top_suspicious['send_sum'].apply(lambda x: f"{x:,.2f}")
        top_suspicious['recv_sum'] = top_suspicious['recv_sum'].apply(lambda x: f"{x:,.2f}")
        
        return top_suspicious, f"已识别出 {top_n} 个最可疑账户"
        
    except Exception as e:
        return None, f"批量预测失败: {str(e)}"

# ===================== Gradio 界面 =====================

def create_ui():
    with gr.Blocks(title="AML 反洗钱检测系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏦 AML 反洗钱检测系统
        
        基于机器学习的可疑账户识别与解释平台
        """)
        
        with gr.Tab("📁 数据导入"):
            gr.Markdown("### 上传 AMLsim 数据文件")
            
            with gr.Row():
                with gr.Column():
                    trans_file = gr.File(
                        label="交易数据 (transactions.csv)",
                        file_types=[".csv"]
                    )
                with gr.Column():
                    acc_file = gr.File(
                        label="账户数据 (accounts.csv)",
                        file_types=[".csv"]
                    )
            
            load_btn = gr.Button("加载数据", variant="primary")
            
            data_info = gr.Markdown()
            
            with gr.Row():
                trans_cols = gr.Textbox(label="交易数据列名", lines=2, interactive=False)
                acc_cols = gr.Textbox(label="账户数据列名", lines=2, interactive=False)
        
        with gr.Tab("🔍 数据探索") as explore_tab:
            gr.Markdown("### 探索数据分布")
            
            with gr.Row():
                with gr.Column():
                    data_type = gr.Radio(
                        choices=["交易数据", "账户数据"],
                        value="交易数据",
                        label="选择数据类型"
                    )
                    n_rows = gr.Slider(5, 50, value=10, step=5, label="显示行数")
                    sample_btn = gr.Button("查看样本")
                
                with gr.Column():
                    sample_output = gr.DataFrame(label="数据样本")
            
            gr.Markdown("### 数据分布可视化")
            viz_btn = gr.Button("生成分布图", variant="primary")
            viz_output = gr.Image(label="数据分布")
            viz_status = gr.Textbox(label="状态", interactive=False)
        
        with gr.Tab("🤖 模型训练") as train_tab:
            gr.Markdown("### 配置模型参数")
            
            with gr.Row():
                with gr.Column():
                    n_estimators = gr.Slider(50, 300, value=100, step=50, label="树的数量")
                    max_depth = gr.Slider(5, 30, value=10, step=5, label="最大深度 (0=无限制)")
                with gr.Column():
                    min_samples_split = gr.Slider(2, 10, value=2, step=1, label="最小分裂样本数")
                    test_size = gr.Slider(0.1, 0.4, value=0.3, step=0.05, label="测试集比例")
            
            use_network = gr.Checkbox(value=True, label="使用网络拓扑特征")
            
            train_btn = gr.Button("开始训练", variant="primary")
            
            train_result = gr.Markdown()
            
            with gr.Row():
                cm_output = gr.Image(label="混淆矩阵")
                fi_output = gr.Image(label="特征重要性")
        
        with gr.Tab("🎯 预测分析") as predict_tab:
            gr.Markdown("### 单账户预测")
            
            with gr.Row():
                with gr.Column():
                    account_id_input = gr.Number(label="输入账户ID", precision=0)
                    predict_btn = gr.Button("预测", variant="primary")
                with gr.Column():
                    predict_result = gr.Markdown()
            
            with gr.Row():
                with gr.Column():
                    explain_btn = gr.Button("生成SHAP解释", variant="secondary")
                with gr.Column():
                    explain_output = gr.Image(label="SHAP瀑布图")
                    explain_status = gr.Textbox(label="状态", interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("### 批量预测最可疑账户")
            
            with gr.Row():
                top_n = gr.Slider(5, 50, value=20, step=5, label="显示前N个可疑账户")
                batch_btn = gr.Button("批量预测", variant="primary")
            
            batch_output = gr.DataFrame(label="可疑账户列表")
            batch_status = gr.Textbox(label="状态", interactive=False)
        
        # 事件绑定
        load_btn.click(
            fn=load_data,
            inputs=[trans_file, acc_file],
            outputs=[data_info, trans_cols, acc_cols, explore_tab]
        )
        
        sample_btn.click(
            fn=show_data_sample,
            inputs=[data_type, n_rows],
            outputs=[sample_output]
        )
        
        viz_btn.click(
            fn=plot_data_distribution,
            inputs=[],
            outputs=[viz_output, viz_status]
        )
        
        train_btn.click(
            fn=train_model,
            inputs=[n_estimators, max_depth, min_samples_split, test_size, use_network],
            outputs=[cm_output, fi_output, train_result, predict_tab]
        )
        
        predict_btn.click(
            fn=predict_account,
            inputs=[account_id_input],
            outputs=[predict_result, explain_btn]
        )
        
        explain_btn.click(
            fn=explain_prediction,
            inputs=[account_id_input],
            outputs=[explain_output, explain_status]
        )
        
        batch_btn.click(
            fn=batch_predict,
            inputs=[top_n],
            outputs=[batch_output, batch_status]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
