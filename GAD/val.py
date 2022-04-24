import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertForTokenClassification
from transformers import get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm
from transformers import Trainer
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, recall_score


def ner_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=18,
                                                       output_attentions=False,
                                                       output_hidden_states=False)

    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    inputs = tokenizer.encode(
        "Hello, my dog is cute",
        add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
        # max_length=160,  # 设定最大文本长度
        pad_to_max_length=True,  # pad到最大的长度
        truncation=True,
        return_tensors='pt'  # 返回的类型为pytorch tensor
    )
    labels = torch.tensor([1] * inputs.size(1)).unsqueeze(0)  # Batch size 1

    outputs = model(inputs, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
    print("labels:", labels)
    print("labels shape:", labels.shape)


def pred_result():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    model_name = "bert-base-cased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Function to get token ids for a list of texts
    def encode_fn(text_list):
        all_input_ids = []
        for text in text_list:
            input_ids = tokenizer.encode(
                text,
                add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
                max_length=160,  # 设定最大文本长度
                pad_to_max_length=True,  # pad到最大的长度
                truncation=True,
                return_tensors='pt'  # 返回的类型为pytorch tensor
            )
            all_input_ids.append(input_ids)
        all_input_ids = torch.cat(all_input_ids, dim=0)
        return all_input_ids

    pd1 = pd.read_excel("./GAD_test.xlsx")
    text_values = pd1['sentence'].tolist()
    all_input_ids = encode_fn(text_values)

    # loadData_val
    # Split data into train and validation
    dataset = TensorDataset(all_input_ids)
    val_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    pred_label = []
    model = torch.load("./gad_model.pkl").to(device)
    model.eval()
    for i, batch in enumerate(val_dataloader):
        with torch.no_grad():
            outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device))
            logits = outputs.get('logits')

            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()
            pred_label.extend(preds)

    pd.DataFrame(data=pred_label, index=range(len(pred_label))).to_csv('pred.csv')


def flat_accuracy(preds, labels):
    """A function for calculating accuracy scores"""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


def compute_f1():
    pd1 = pd.read_excel("./GAD_test.xlsx")
    label_values = pd1['label'].tolist()
    pred_values = pd1['pred'].tolist()
    acc = accuracy_score(label_values, pred_values)
    f1 = f1_score(label_values, pred_values)
    recall = recall_score(label_values, pred_values)
    print(acc)
    print(f1)
    print(recall)


if __name__ == '__main__':
    # compute_f1()
    ner_bert()
