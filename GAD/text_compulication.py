import pandas as pd

# loadData_train
pd1 = pd.read_excel("./GAD_train.xlsx")
print(len(pd1))
df1 = pd1.drop_duplicates(subset=['sentence', 'label'], keep='first')
print(len(df1))

