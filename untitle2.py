import os
import pandas as pd

data_dir = r"E:\Document\毕业论文——张清祥\AMLsim数据集\10Kvertices-1Medges\10Kvertices-1Medges"
print("文件夹中的文件：", os.listdir(data_dir))

if 'accounts.csv' in os.listdir(data_dir):
    acc = pd.read_csv(os.path.join(data_dir, 'accounts.csv'), nrows=5)
    print("accounts.csv 列名：", acc.columns.tolist())
    print(acc.head())

# 读取完整数据（如果文件很大可先读一部分，但 1 万到 10 万笔通常内存足够）
df = pd.read_csv(os.path.join(data_dir, 'transactions.csv'))
print(f"总交易笔数：{len(df)}")
print(f"洗钱交易占比：{df['IS_FRAUD'].mean():.4%}")
print(df['IS_FRAUD'].value_counts())

print("TIMESTAMP 最小值：", df['TIMESTAMP'].min())
print("TIMESTAMP 最大值：", df['TIMESTAMP'].max())