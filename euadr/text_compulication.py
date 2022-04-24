import pandas as pd

# loadData_train
df1 = pd.read_excel("./euadr_train.xlsx")
df2 = pd.read_excel("./euadr_test.xlsx")
res = pd.concat([df1, df2], axis=0, ignore_index=True)
print(len(res))
new_res = res.drop_duplicates(subset=['sentence', 'label'], keep='first')
print(len(df1))
new_res.to_csv('./euadr_drop_duplicates.csv', encoding='utf-8')
