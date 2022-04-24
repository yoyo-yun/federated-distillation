import pandas as pd

df0 = pd.read_csv("./i2b2.csv")
df1 = pd.read_csv("./i2b2_test.csv")
res = pd.concat([df0,df1], axis=0, ignore_index=True)
res.to_csv('./i2b2_all.csv', encoding='utf-8')