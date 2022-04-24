from transformers import AutoTokenizer, AutoModelForMaskedLM, BertConfig, AutoModel, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm
from transformers import Trainer
import torch.nn.functional as F
import VID

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
cuda_index = "cuda:0"
# cpu_index = 'cpu'
teacher_device = 'cuda:0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device(cuda_index)
model_name = "microsoft/MiniLM-L12-H384-uncased"

# 构建数据集
epochs = 80
batch_size = 32
T = 3
alpha = 0.8
lambda_value = 0.6
C = 0.7

center_model_name = "MiniLM"
Cache_file = './working_dir'
model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=Cache_file)
# model.classifier = nn.Sequential(nn.Linear(768,151),nn.Linear(151,2),nn.Softmax())
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=Cache_file)
model = model.to(cuda_index)
# teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_name, cache_dir=Cache_file)
# teacher_model.load_state_dict(
#     torch.load("./center_model/center_model_bio_bert.pth", map_location='cuda:0'))
# teacher_model_name_1 = 'albert'
# teacher_model_name_2 = 'scibert'
# teacher_model_name_3 = 'PubMedBERT'
teacher_model_1 = torch.load("./clients_model/clients_model_scibert.pkl", map_location=teacher_device)
teacher_model_2 = torch.load("./clients_model/clients_model_roberta.pkl", map_location=teacher_device)
# teacher_model_3 = torch.load("./clients_model/clients_model_ClinicalBERT.pkl", map_location=teacher_device)
# teacher_model_4 = torch.load("./clients_model/clients_model_biobert.pkl", map_location=teacher_device)
teacher_model_5 = torch.load("./clients_model/clients_model_bert.pkl", map_location=teacher_device)
teacher_model_6 = torch.load("./clients_model/clients_model_Bio-renet.pkl", map_location=teacher_device)
teacher_model_list = [teacher_model_1, teacher_model_2, teacher_model_5,
                      teacher_model_6]
C_index = int(len(teacher_model_list) * C)


# teacher_model_name = teacher_model_name_1 + teacher_model_name_2 + teacher_model_name_3


# teacher_model = teacher_model.to(cuda_index)


# Function to get token ids for a list of texts
def encode_fn(text_list):
    all_input_ids = []
    for text in text_list:
        input_ids = tokenizer.encode(
            text,
            add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
            max_length=60,  # 设定最大文本长度
            pad_to_max_length=True,  # pad到最大的长度
            truncation=True,
            return_tensors='pt'  # 返回的类型为pytorch tensor
        )
        all_input_ids.append(input_ids)
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids


# df0 = pd.read_csv("./shuffleDatasets/data_0.csv")
# df1 = pd.read_csv("./shuffleDatasets/data_1.csv")
# df2 = pd.read_csv("./shuffleDatasets/data_2.csv")
# df3 = pd.read_csv("./shuffleDatasets/data_3.csv")
# df4 = pd.read_csv("./shuffleDatasets/data_4.csv")
# df5 = pd.read_csv("./shuffleDatasets/data_5.csv")
# df6 = pd.read_csv("./shuffleDatasets/data_6.csv")
df7 = pd.read_csv("./shuffleDatasets/data_7.csv")
df8 = pd.read_csv("./shuffleDatasets/data_8.csv")

res = pd.concat([df7, df8], axis=0, ignore_index=True)
labels = res['label'].tolist()
text_values = res['sentence'].tolist()
all_input_ids = encode_fn(text_values)
labels = torch.tensor(labels)
dataset = TensorDataset(all_input_ids, labels)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# loadData val
pd1 = pd.read_csv("./shuffleDatasets/data_9.csv")
labels_list_val = pd1['label'].tolist()
text_values_val = pd1['sentence'].tolist()
all_input_ids_val = encode_fn(text_values_val)
labels_val = torch.tensor(labels_list_val)
dataset_val = TensorDataset(all_input_ids_val, labels_val)
val_dataloader = DataLoader(dataset_val, batch_size=60, shuffle=False)

# create optimizer and learning rate schedule
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


def flat_accuracy(preds, labels):
    """A function for calculating accuracy scores"""

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat), precision_score(labels_flat, pred_flat), \
           recall_score(labels_flat, pred_flat), f1_score(labels_flat, pred_flat)


# vid loss
vid_loss = VID.VIDLoss(3, 3, 2, device=cuda_index)


class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, logits_teacher, vid_loss, return_outputs=True):
        # labels = torch.nn.functional.one_hot(inputs[1], 25).to(device)
        model = model.to(cuda_index)
        outputs = model(inputs[0].to(cuda_index))
        logits = outputs.get('logits')

        labels = inputs[1].to(cuda_index)

        # CE loss self-student
        loss_nll = F.cross_entropy(
            logits.view(-1, 2), labels.squeeze(-1),
            reduction='mean')

        # logits = F.softmax(logits)
        # logits_teacher = F.softmax(logits_teacher)
        d_loss = nn.KLDivLoss(reduction='mean')(F.log_softmax(logits.view(-1, 2) / T, dim=1),
                                                F.softmax(logits_teacher.view(-1, 2) / T,
                                                          dim=1)) * T * T
        d_loss = d_loss.mean()
        vid_loss = vid_loss(F.softmax(logits.view(-1, 2), dim=1),
                            F.softmax(logits_teacher.view(-1, 2), dim=1), 'mean')

        # loss_all = (alpha) * loss_nll + ((1 - alpha) / 2) * d_loss + ((1 - alpha) / 2) * vid_loss
        loss_all = (alpha) * loss_nll + (1 - alpha) * d_loss + lambda_value * vid_loss
        return (loss_all, outputs) if return_outputs else loss


def stacking(logits_list):
    teacher_logits = torch.zeros_like(logits_list[0])
    logits_num = len(logits_list)
    single_length = int(len(teacher_logits) / logits_num)
    for i, temp_logits in enumerate(logits_list):
        if i + 1 < logits_num:
            teacher_logits[i * single_length:i * single_length + single_length] = temp_logits[
                                                                                  i * single_length:i * single_length + single_length]
            # print("===========logits 1===========")
            # print(temp_logits[i * single_length:i * single_length + single_length])
        else:
            teacher_logits[i * single_length:] = temp_logits[i * single_length:]
            # print("===========logits 2===========")
            # print(temp_logits[i * single_length:])

    return teacher_logits


def vote(logits_list):
    teacher_logits = torch.zeros_like(logits_list[0])
    for i, temp_logits in enumerate(logits_list):
        # print("======torch.argmax_logits======")
        arg_logits = torch.argmax(temp_logits, dim=-1)
        arg_logits = torch.nn.functional.one_hot(arg_logits, 2)
        # print(arg_logits)
        teacher_logits += arg_logits

    return teacher_logits


def teacher_train(batch, teacher_model_list):
    logits_list = []
    index_list = []
    for i in range(100):
        index = random.randint(0, len(teacher_model_list))
        index_list.append(index)
        index_list = list(set(index_list))
        if len(index_list) > C_index:
            break
    for index in index_list:
        # index = random.randint(0,3)
        # print(index)
        outputs_teacher_1 = teacher_model_list[index](batch[0].to(teacher_device))
        logits_teacher_1 = outputs_teacher_1.get('logits')
        logits_list.append(logits_teacher_1)

    # logits_teacher = vote(logits_list)
    logits_teacher = vote(logits_list)
    return logits_teacher


multilabelTrainer = MultilabelTrainer(model.to(cuda_index))
best_f1 = 0
best_p = 0
best_r = 0
for epoch in range(epochs):
    model.train()
    total_loss, total_val_loss = 0, 0
    total_eval_accuracy = 0
    total_eval_precision = 0
    total_eval_recall = 0
    total_eval_f1 = 0
    for step, batch in tqdm(enumerate(train_dataloader)):
        model.zero_grad()
        # teacher_model logits
        logits_teacher = teacher_train(batch, teacher_model_list)
        logits_teacher = logits_teacher.to(cuda_index)

        loss, logits = multilabelTrainer.compute_loss(model, batch, logits_teacher, vid_loss)
        total_loss += loss.item()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    for i, batch in enumerate(val_dataloader):
        with torch.no_grad():
            # loss, logits = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
            #                      labels=batch[1].to(device))
            output = model(batch[0].to(cuda_index))
            logits = output['logits']
            loss = F.cross_entropy(
                logits.view(-1, 2), batch[1].squeeze(-1).to(cuda_index),
                reduction='mean')
            # loss = output['loss']
            # logits = output['logits']
            total_val_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = batch[1].to('cpu').numpy()
            step_eval_accuracy, step_eval_precision, step_eval_recall, step_eval_f1 = flat_accuracy(logits, label_ids)
            total_eval_accuracy += step_eval_accuracy
            total_eval_precision += step_eval_precision
            total_eval_recall += step_eval_recall
            total_eval_f1 += step_eval_f1

    avg_train_loss = total_loss / len(train_dataloader)
    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    avg_val_precision = total_eval_precision / len(val_dataloader)
    avg_val_recall = total_eval_recall / len(val_dataloader)
    avg_val_f1 = total_eval_f1 / len(val_dataloader)

    print("This is epoch :" + str(epoch))
    print(f'Train loss     : {avg_train_loss}')
    print(f'Validation loss: {avg_val_loss}')
    print(f'Accuracy: {avg_val_accuracy:.4f}')
    print(f'Precision: {avg_val_precision:.4f}')
    print(f'Recall: {avg_val_recall:.4f}')
    print(f'F1  : {avg_val_f1:.4f}')
    if best_f1 < avg_val_f1:
        best_f1 = avg_val_f1
        best_p = avg_val_precision
        best_r = avg_val_recall
        print(f"save model best F1 : {best_f1}")
        torch.save(model, './FD_model/distill_center_model_' + center_model_name + '.pkl')
        torch.save(model.state_dict(), './FD_model/distill_center_model_' + center_model_name + '.pth')
    # print("{" + teacher_model_name + "}")
    print("T:{" + str(T) + "} alpha:" + str(alpha) + " lambda_value:{" + str(lambda_value) + "}")
    print("Best  F1:{" + str(best_f1) + "} Precision:" + str(best_p) + " Recall:{" + str(best_r) + "}")

    print('\n')
