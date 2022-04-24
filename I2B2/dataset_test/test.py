# coding:utf-8
import fileinput
import os
import pandas as pd
import re

p = re.compile("\"(.*?)\"")


def processor(filename, is_test):
    if is_test == "test":
        path = r"./txt_test/" + filename
        rel_path = r"./rel_test/" + filename.split("txt")[0] + "rel"
    else:
        path = r"./txt/" + filename
        rel_path = r"./rel/" + filename.split("txt")[0] + "rel"
    text = []
    for line in fileinput.input(path):
        # print(line)
        text.append(line)
    # print(len(text))
    rel_index = []
    rel = []
    con_list_1 = []
    con_list_2 = []
    for line in fileinput.input(rel_path):
        b = line.split(" ")
        for i in [b[-1]]:
            if ":" in i:
                rel_index.append(i[:i.index(":")])
        b = line.split("||")
        for i in [b[1]]:
            if "r=" in i:
                rel.append(i[i.index("=") + 1:].replace("\"", ""))
        conce_list = p.findall(line)
        con_list_1.append(conce_list[0])
        con_list_2.append(conce_list[2])
    train_label = []
    for i in range(len(text)):
        train_label.append("Other")
    for i, index in enumerate(rel_index):
        train_label[int(index) - 1] = rel[i]
        text[int(index) - 1] = con_list_1[i] + " " + con_list_2[i] + " " + text[int(index) - 1]
    # print(rel_index)
    # print(rel)
    # print(train_label)
    return text, train_label


def huizong_excel(col_name, sum_list, flag):
    data_new = []

    for i in range(0, len(sum_list[0])):
        temp_data = []
        for j_list in sum_list:
            temp_data.append(j_list[i])

        data_new.append(temp_data)

    recruit_data_new = pd.DataFrame(
        columns=col_name,
        data=data_new,
    )
    recruit_data_new = recruit_data_new.sample(frac=1)
    recruit_data_new.drop_duplicates(subset=['sentence'], keep='first', inplace=True)
    print(len(recruit_data_new))
    recruit_data_new.to_csv('./' + str(flag) + '.csv', encoding='utf-8')


# 要查找的目录
file_dir = './txt'
# 目录下面的所有文件名
os.listdir(file_dir)
text = []
label = []
for i in os.listdir(file_dir):
    # print(i)
    temp_text, temp_train_label = processor(i, "no")
    text += temp_text
    label += temp_train_label
label_index = list(sorted(set(label)))
print(label_index)
tag_index = []
for i in label:
    tag_index.append(label_index.index(i))
print(len(label_index))
print(len(set(text)))
huizong_excel(['sentence', 'label'], [text, tag_index], "i2b2_train")
# print(len(text))
# print(len(label))
# print(set(label))
