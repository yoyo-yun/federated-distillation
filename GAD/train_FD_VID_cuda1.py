from transformers import AutoTokenizer, AutoModelForMaskedLM,BertConfig
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm
from transformers import Trainer
from sklearn.metrics import f1_score, accuracy_score, recall_score
import VID

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0')
model_name = 'emilyalsentzer/Bio_ClinicalBERT'

tokenizer = BertTokenizer.from_pretrained('../Bio_ClinicalBERT/vocab.txt')
config = BertConfig.from_pretrained('../Bio_ClinicalBERT/config.json')
model = BertForSequenceClassification.from_pretrained('../Bio_ClinicalBERT/pytorch_model.bin', config=config)

# Load the pretrained BERT model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
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


# 构建数据集
epochs = 20
batch_size = 20

# loadData train
pd1 = pd.read_csv("./shuffleDatasets/data_0.csv")
labels = pd1['label'].tolist()
text_values = pd1['sentence'].tolist()
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

teacher_model0 = torch.load("./clients_model/model3_bio_bert.pkl").to('cpu')
teacher_model1 = torch.load("./clients_model/model5_bio_bert.pkl").to('cpu')
teacher_model2 = torch.load("./clients_model/model6_bio_bert.pkl").to('cpu')
teacher_model3 = torch.load("./clients_model/model8_bio_bert.pkl").to('cpu')
# create optimizer and learning rate schedule
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


def flat_accuracy(preds, labels):
    """A function for calculating accuracy scores"""

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat), recall_score(labels_flat, pred_flat), f1_score(labels_flat,
                                                                                                  pred_flat)


# vid loss
vid_loss = VID.VIDLoss(3, 3, 2, device=device)

T = 5
alpha = 0.7


class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, logits_teacher, vid_loss, return_outputs=True):
        # labels = torch.nn.functional.one_hot(inputs[1], 25).to(device)
        model = model.to(device)
        outputs = model(inputs[0].to(device), token_type_ids=None, attention_mask=(inputs[0] > 0).to(device))
        logits = outputs.get('logits')

        labels = inputs[1].to(device)

        # CE loss self-student
        loss_nll = F.cross_entropy(
            logits.view(-1, self.model.config.num_labels), labels.squeeze(-1),
            reduction='mean')

        d_loss = nn.KLDivLoss(reduction='mean')(F.log_softmax(logits.view(-1, self.model.config.num_labels) / T, dim=1),
                                                F.softmax(logits_teacher.view(-1, self.model.config.num_labels) / T,
                                                          dim=1)) * T * T
        d_loss = d_loss.mean()

        vid_loss = vid_loss(F.softmax(logits.view(-1, self.model.config.num_labels), dim=1),
                            F.softmax(logits_teacher.view(-1, self.model.config.num_labels), dim=1), 'mean')

        # loss_all = (alpha) * loss_nll + ((1 - alpha) / 2) * d_loss + ((1 - alpha) / 2) * vid_loss
        loss_all = (alpha) * loss_nll + (1 - alpha) * d_loss + 0.15 * vid_loss
        return (loss_all, outputs) if return_outputs else loss


multilabelTrainer = MultilabelTrainer(model)
best_f1 = 0
for epoch in range(epochs):
    model.train()
    total_loss, total_val_loss = 0, 0
    total_eval_accuracy = 0
    total_eval_recall = 0
    total_eval_f1 = 0
    for step, batch in tqdm(enumerate(train_dataloader)):
        model.zero_grad()
        # teacher_model logits
        outputs_teacher0 = teacher_model0(batch[0], token_type_ids=None, attention_mask=(batch[0] > 0))
        logits_teacher0 = outputs_teacher0.get('logits')
        outputs_teacher1 = teacher_model1(batch[0], token_type_ids=None, attention_mask=(batch[0] > 0))
        logits_teacher1 = outputs_teacher1.get('logits')
        outputs_teacher2 = teacher_model2(batch[0], token_type_ids=None, attention_mask=(batch[0] > 0))
        logits_teacher2 = outputs_teacher2.get('logits')
        outputs_teacher3 = teacher_model3(batch[0], token_type_ids=None, attention_mask=(batch[0] > 0))
        logits_teacher3 = outputs_teacher3.get('logits')
        logits_teacher = (logits_teacher0 + logits_teacher1 + logits_teacher2 + logits_teacher3) / 4
        logits_teacher = logits_teacher.to(device)

        loss, logits = multilabelTrainer.compute_loss(model, batch, logits_teacher, vid_loss)

        total_loss += loss.item()

        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    for i, batch in enumerate(val_dataloader):
        with torch.no_grad():
            output = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                           labels=batch[1].to(device))

            loss = output['loss']
            logits = output['logits']
            total_val_loss += loss.item()

            logits = logits.detach().cpu().numpy()
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
        torch.save(model, './FD_model/GAD_model_bio_bert.pkl')
    print('\n')
