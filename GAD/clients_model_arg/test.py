from transformers import AutoTokenizer, AutoModelForMaskedLM, BertConfig
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

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0')
model_name = 'emilyalsentzer/Bio_ClinicalBERT'

tokenizer = BertTokenizer.from_pretrained('../../bio-bert/vocab.txt')
config = BertConfig.from_pretrained('../../bio-bert/config.json')
model = BertForSequenceClassification.from_pretrained('../../bio-bert/pytorch_model.bin', config=config)