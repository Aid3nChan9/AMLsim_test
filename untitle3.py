import pandas as pd

data_dir = r"E:\Document\毕业论文——张清祥\AMLsim数据集\10Kvertices-1Medges\10Kvertices-1Medges"

# 读取账户表
acc = pd.read_csv(data_dir + "/accounts.csv")
print("账户总数：", len(acc))
print("可疑账户数量：", acc['IS_FRAUD'].sum())
print("可疑账户占比：{:.2%}".format(acc['IS_FRAUD'].mean()))
print(acc['IS_FRAUD'].value_counts())