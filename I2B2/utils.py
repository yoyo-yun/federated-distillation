import numpy as np
import pandas as pd
import re
import torch
# from keras.layers import Input, Dense, LSTM, CuDNNLSTM, Dropout, Embedding, Softmax, CuDNNGRU
# from keras.layers import Bidirectional, concatenate, RepeatVector, Dot, Activation, merge, Reshape, Add
# from keras.models import Model, Sequential
# from keras.optimizers import Adam
# from keras import backend as K
# from keras.callbacks import TensorBoard
# from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# # from utils import f1, get_train, clean_str, load_glove
# import tensorflow as tf
import os  # for setting GPU
import time  # for formatting log name


def get_train(file_path, no_dup=True, no_other=False):
    """
    format the i2b2 data into a pandas dataframe.
    notice that there are many duplicates texts in the dataset, so adjust the parameters according
    to your interests.

    parameters:
        file_path: the file's path, a string format
        no_dup: if true, the duplicate text would be removed
        no_other: if true, the samples of tag "other" should be removed

    sample usage: train_df = get_train("./training file.txt")
    return : a pd dataframe with columns: text, tag, test_info, problem_info, treatment_info
    """

    file = open(file_path)
    file = [line.strip('\n').strip('\ufeff') for line in file.readlines()]

    def format_input(df):
        targets = ['test', 'problem', 'treatment']
        for target in targets:
            df.loc[df['t1'].str.contains('\|' + target), target + '_info'] = df['t1']
            df.loc[(df['t2'].str.contains('\|' + target)) & \
                   (df[target + '_info'].isnull()), target + '_info'] = df['t2']
        df.drop(['t1', 't2'], axis=1, inplace=True)
        if no_dup:
            df.drop_duplicates(['text'], inplace=True)
        if no_other:
            df = df.loc[df.tag != 'other']  # delete tag "other"
        df.index = np.arange(df.shape[0])
        return df

    train_df = pd.DataFrame(np.array([file[i::5] for i in range(4)]).T, columns=['text', 't1', 't2', 'tag'])
    train_df = format_input(train_df)
    return train_df


def clean_str(text, lower=True):
    """
    clean and format the text

    parameters:
        text: a string format text
        lower: if true, the text would be convert to lower format
    
    return: processed text
    """

    text = text.lower()

    replace_pair = [(r"[^A-Za-z0-9^,!.\/'+-=]", " "), (r"what's", "what is "), (r"that's", "that is "),
                    (r"there's", "there is "),
                    (r"it's", "it is "), (r"\'s", " "), (r"\'ve", " have "), (r"can't", "can not "), (r"n't", " not "),
                    (r"i'm", "i am "),
                    (r"\'re", " are "), (r"\'d", " would "), (r"\'ll", " will "), (r",", " "), (r"\.", " "),
                    (r"!", " ! "), (r"\/", " "),
                    (r"\^", " ^ "), (r"\+", " + "), (r"\-", " - "), (r"\=", " = "), (r"'", " "),
                    (r"(\d+)(k)", r"\g<1>000"), (r":", " : "),
                    (r" e g ", " eg "), (r" b g ", " bg "), (r" u s ", " american "), (r"\0s", "0"), (r" 9 11 ", "911"),
                    (r"e - mail", "email"),
                    (r"j k", "jk"), (r"\s{2,}", " ")]

    for before, after in replace_pair:
        text = re.sub(before, after, text)

    return text.strip()


# def load_glove(word_index):
#     def get_coefs(word,*emb): return word, np.asarray(emb,dtype='float32')
#     embedding = dict(get_coefs(*o.split(' ')) for o in tqdm(open('glove.840B.300d.txt')))
#
#     emb_mean, emb_std = -0.005838459, 0.48782179
#     embed_matrix = np.random.normal(emb_mean,emb_std,(max_features,emb_size))
#
#     for word, i in word_index.items():
#         if i >= max_features: continue
#         if embedding.get(word) is not None:
#             embed_matrix[i] = embedding.get(word)
#
#     return embed_matrix
#
# def f1(y_true, y_pred):
#     def recall(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#         recall = true_positives / (possible_positives + K.epsilon())
#         return recall
#
#     def precision(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + K.epsilon())
#         return precision
#
#     precision = precision(y_true, y_pred)
#     recall = recall(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))
a = get_train("./training file.txt")
tag_list = a['tag'].to_list()
text_list = a['text'].to_list()
labels_index = ['PIP', 'TeCP', 'TeRP', 'TrAP', 'TrCP', 'other']
tag_index = []
for i in tag_list:
    tag_index.append(labels_index.index(i))

targets = ['test_info', 'problem_info', 'treatment_info']
text = []
for i in range(0, len(text_list)):
    temp_str = text_list[i]
    # if str(a['test_info'][i]) != "nan":
    #     temp_str += " " + str(a['test_info'][i])
    # if str(a['problem_info'][i]) != "nan":
    #     temp_str += " " + str(a['problem_info'][i])
    # if str(a['treatment_info'][i]) != "nan":
    #     temp_str += " " + str(a['treatment_info'][i])
    # print(str(a['test_info'][i])+" "+str(a['problem_info'][i])+" "+str(a['treatment_info'][i]))
    print(temp_str)
    text.append(temp_str)


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
    recruit_data_new = recruit_data_new.sample(frac=1)
    recruit_data_new.to_csv('./' + str(flag) + '.csv', encoding='utf-8')


huizong_excel(len(text), ['sentence', 'label'], [text, tag_index], "i2b2_only_text")
# torch.nn.functional.one_hot()
