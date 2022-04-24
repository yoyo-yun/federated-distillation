from transformers import AutoTokenizer, AutoModelForMaskedLM, BertConfig, AutoModel, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, AlbertForSequenceClassification, AlbertTokenizer, \
    AlbertConfig
from transformers import get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm
from transformers import Trainer
import torch.nn.functional as F
import os

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
cuda_index = "cuda:0"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device(cuda_index)
# 构建数据集
epochs = 90
batch_size = 128
average = "micro"
class_num = 9
# F1:83
# model_name = "jambo/microsoftBio-renet"

# model_name = 'microsoft/MiniLM-L12-H384-uncased'

# fd_mode_name = ['jambo/microsoftBio-renet', 'bert-base-uncased', 'dmis-lab/biobert-base-cased-v1.2',
#                 'emilyalsentzer/Bio_ClinicalBERT', 'roberta-base', 'allenai/scibert_scivocab_uncased']
# fd_mode_name_list = ['Bio-renet', 'bert', 'biobert',
#                      'ClinicalBERT', 'roberta', 'scibert']

# 自动导入模型
Cache_file = '../GAD/working_dir'
file_index = 0
fd_mode_name = ['allenai/scibert_scivocab_uncased']
fd_mode_name_list = ['scibert']
for model_name in fd_mode_name:
    # model = BertForSequenceClassification.from_pretrained(model_name, cache_dir=Cache_file,
    #                                                       attention_probs_dropout_prob=0.3)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=Cache_file, num_labels=class_num)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=Cache_file)
    # model.classifier = nn.Sequential(nn.Linear(768, 151), nn.Linear(151, class_num), nn.Softmax())
    model.to(cuda_index)


    # Function to get token ids for a list of texts
    def encode_fn(text_list):
        all_input_ids = []
        for text in text_list:
            input_ids = tokenizer.encode(
                text,
                add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
                max_length=80,  # 设定最大文本长度
                pad_to_max_length=True,  # pad到最大的长度
                truncation=True,
                return_tensors='pt'  # 返回的类型为pytorch tensor
            )
            all_input_ids.append(input_ids)
        all_input_ids = torch.cat(all_input_ids, dim=0)
        return all_input_ids


    df0 = pd.read_csv("./shuffleDatasets/data_" + str(4) + ".csv")
    df1 = pd.read_csv("./shuffleDatasets/data_" + str(5) + ".csv")
    print("==========model name============")
    print("==========" + model_name + "============")
    print("==========train Dataset============")
    print("==========" + str(2) + " and " + str(3) + "============")

    res = pd.concat([df0, df1], axis=0, ignore_index=True)
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

    # Load the pretrained BERT model

    # create optimizer and learning rate schedule

    # optimizer = torch.optim.SGD(params=model.parameters(), lr=2e-5, momentum=0.9, dampening=0.5, weight_decay=0.01, nesterov=False)
    lr = 3e-06
    # optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    # warm_up_ratio = 0.1  # 定义要预热的step
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_ratio * total_steps,
    #                                             num_training_steps=total_steps)

    from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


    def flat_accuracy(preds, labels):
        """A function for calculating accuracy scores"""

        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return accuracy_score(labels_flat, pred_flat), precision_score(labels_flat, pred_flat, average=average), \
               recall_score(labels_flat, pred_flat, average=average), f1_score(labels_flat, pred_flat, average=average)


    class MultilabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=True):
            # labels = torch.nn.functional.one_hot(inputs[1], 25).to(device)
            # outputs = model(inputs[0].to(device), token_type_ids=None, attention_mask=(inputs[0] > 0).to(device))
            model = model.to(device)
            outputs = model(inputs[0].to(device))
            logits = outputs.get('logits')

            labels = inputs[1].to(device)
            loss = F.cross_entropy(
                logits.view(-1, class_num), labels.squeeze(-1),
                reduction='mean')
            return (loss, outputs) if return_outputs else loss


    multilabelTrainer = MultilabelTrainer(model)
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
            loss, logits = multilabelTrainer.compute_loss(model, batch)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.75)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()
        for i, batch in enumerate(val_dataloader):
            with torch.no_grad():
                loss, logits = multilabelTrainer.compute_loss(model, batch)
                total_val_loss += loss.item()

                logits = logits['logits'].detach().cpu().numpy()
                label_ids = batch[1].to('cpu').numpy()
                step_eval_accuracy, step_eval_precision, step_eval_recall, step_eval_f1 = flat_accuracy(logits,
                                                                                                        label_ids)
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
            torch.save(model.state_dict(), './clients_model/clients_model_' + fd_mode_name_list[file_index] + '.pth')
            torch.save(model, './clients_model/clients_model_' + fd_mode_name_list[file_index] + '.pkl')
        print("F1:{" + str(best_f1) + "} Precision:" + str(best_p) + " Recall:{" + str(best_r) + "}")
        print("{" + fd_mode_name_list[file_index] + "}")
        print('\n')
    file_index += 1
