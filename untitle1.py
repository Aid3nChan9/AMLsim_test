import pandas as pd

# 方法1：使用原始字符串（推荐）
df = pd.read_csv(r"E:\Document\毕业论文——张清祥\AMLsim数据集\10Kvertices-1Medges\10Kvertices-1Medges\transactions.csv", nrows=5)

# 或者方法2：使用正斜杠（Windows也支持）
# df = pd.read_csv("E:/Document/毕业论文——张清祥/AMLsim数据集/10Kvertices-1Medges/10Kvertices-1Medges/transactions.csv", nrows=5)

print("列名：", df.columns.tolist())
print("前5行：")
print(df.head())
