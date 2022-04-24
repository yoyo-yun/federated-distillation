from transformers import AutoTokenizer, AutoModelForMaskedLM,BertConfig
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm
from transformers import Trainer
import torch.nn.functional as F

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:1')
file_num = 9
model_name = 'dmis-lab/biobert-base-cased-v1.2'

tokenizer = BertTokenizer.from_pretrained('../bio-bert/vocab.txt')
config = BertConfig.from_pretrained('../bio-bert/config.json')
model = BertForSequenceClassification.from_pretrained('../bio-bert/pytorch_model.bin', config=config)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# # Load the pretrained BERT model
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2,
#                                                       output_attentions=False,
#                                                       output_hidden_states=False)
model.to(device)


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


# loadData val
pd1 = pd.read_csv("./shuffleDatasets/data_9.csv")
labels_list_val = pd1['label'].tolist()
text_values_val = pd1['sentence'].tolist()
all_input_ids_val = encode_fn(text_values_val)
labels_val = torch.tensor(labels_list_val)
dataset_val = TensorDataset(all_input_ids_val, labels_val)
val_dataloader = DataLoader(dataset_val, batch_size=60, shuffle=False)

# loadData
all_input_ids_list = []
labels_tensor_list = []
for i in range(0, file_num):
    pd1 = pd.read_csv("./shuffleDatasets/data_" + str(i) + ".csv")
    labels_list = pd1['label'].tolist()
    text_values = pd1['sentence'].tolist()
    all_input_ids = encode_fn(text_values)
    labels = torch.tensor(labels_list)
    all_input_ids_list.append(all_input_ids)
    labels_tensor_list.append(labels)

for j in range(0, len(all_input_ids_list)):
    # 构建数据集
    epochs = 5
    batch_size = 15
    best_f1 = 0
    best_p=0
    best_r=0
    # Split data into train and validation
    dataset = TensorDataset(all_input_ids_list[j], labels_tensor_list[j])
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # create optimizer and learning rate schedule
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    from sklearn.metrics import f1_score, accuracy_score, recall_score


    def flat_accuracy(preds, labels):
        """A function for calculating accuracy scores"""

        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return accuracy_score(labels_flat, pred_flat), recall_score(labels_flat, pred_flat), f1_score(labels_flat,
                                                                                                      pred_flat)


    class MultilabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=True):
            # labels = torch.nn.functional.one_hot(inputs[1], 25).to(device)
            model = model.to(device)
            outputs = model(inputs[0].to(device), token_type_ids=None, attention_mask=(inputs[0] > 0).to(device))
            logits = outputs.get('logits')

            labels = inputs[1].to(device)
            loss = F.cross_entropy(
                logits.view(-1, self.model.config.num_labels), labels.squeeze(-1),
                reduction='mean')
            return (loss, outputs) if return_outputs else loss


    multilabelTrainer = MultilabelTrainer(model)
    for epoch in range(epochs):
        model.train()
        total_loss, total_val_loss = 0, 0
        total_eval_accuracy = 0
        total_eval_recall = 0
        total_eval_f1 = 0
        for step, batch in tqdm(enumerate(train_dataloader)):
            model.zero_grad()
            loss, logits = multilabelTrainer.compute_loss(model, batch)

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        for i, batch in enumerate(val_dataloader):
            with torch.no_grad():
                loss, logits = multilabelTrainer.compute_loss(model, batch)
                total_val_loss += loss.item()
                logits = logits['logits'].detach().cpu().numpy()
                label_ids = batch[1].to('cpu').numpy()
                step_eval_accuracy, step_eval_recall, step_eval_f1 = flat_accuracy(logits, label_ids)
                total_eval_accuracy += step_eval_accuracy
                total_eval_recall += step_eval_recall
                total_eval_f1 += step_eval_f1

        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        avg_val_recall = total_eval_recall / len(val_dataloader)
        avg_val_f1 = total_eval_f1 / len(val_dataloader)

        print("This is epoch :" + str(epoch))
        print(f'Train loss     : {avg_train_loss}')
        print(f'Validation loss: {avg_val_loss}')
        print(f'Accuracy: {avg_val_accuracy:.4f}')
        print(f'Recall: {avg_val_recall:.4f}')
        print(f'F1  : {avg_val_f1:.4f}')
        if best_f1 < avg_val_f1:
            best_f1 = avg_val_f1
            print(f"save model best F1 : {best_f1}")
            # torch.save(model.state_dict(), './clients_model/model' + str(j) + '_bio_bert.pkl')
            torch.save(model.state_dict(), './clients_model_arg/model' + str(j) + '_bio_bert.pth')
        print('\n')

