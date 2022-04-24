import math
import os
import pandas as pd
from tqdm.autonotebook import tqdm


def huizong_excel(qy_num, col_name, sum_list, flag):
    data_new = []

    for i in range(0, qy_num):
        temp_data = []
        for j_list in sum_list:
            temp_data.append(j_list[i])

        data_new.append(temp_data)

    recruit_data_new = pd.DataFrame(
        columns=col_name,
        data=data_new,
    )
    recruit_data_new.to_csv('./' + str(flag) + '.csv', encoding='utf-8')


def merger_text():
    rankings_colname = ['data', 'class']
    data = []
    num_class = []

    for i in range(1, 11):
        a = pd.read_csv("./" + str(i) + "/train.tsv", sep='\t', header=None, names=rankings_colname)
        data += a['data'].tolist()
        num_class += a['class'].tolist()
        b = pd.read_csv("./" + str(i) + "/test.tsv", sep='\t')
        data += b['sentence'].tolist()
        num_class += b['label'].tolist()

    print(len(data))
    print(len(num_class))

    col_name = ['sentence', 'label']
    sum_list = [data, num_class]
    huizong_excel(len(data), col_name, sum_list, "new_GAD_train")


def shuffle_datasets():
    pd1 = pd.read_csv("./i2b2_all.csv")
    pd1 = pd1.drop_duplicates(subset=['sentence', 'label'], keep='first')
    pd1 = pd1.sample(frac=1).reset_index(drop=False)
    n = 10  # 切割份数
    df = pd1
    df_num = len(df)
    print(f"df is length is {df_num}")
    every_epoch_num = math.floor((df_num / n))
    for index in tqdm(range(n)):
        file_name = f'./shuffleDatasets/data_{index}.csv'
        if index < n - 1:
            df_tem = df[every_epoch_num * index: every_epoch_num * (index + 1)]
        else:
            df_tem = df[every_epoch_num * index:]
        print(f'./shuffleDatasets/data_{index}.csv : length is {len(df_tem)}')
        df_tem.to_csv(file_name, index=False)


if __name__ == '__main__':
    shuffle_datasets()
