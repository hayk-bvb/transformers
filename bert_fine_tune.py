import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

from tqdm import tqdm, trange

import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt

# Mac CPU
device = torch.device("cpu")

# ---- Loading the dataset ----
# source of the dataset : https://nyu-mll.github.io/CoLA
df = pd.read_csv("in_domain_train.tsv", delimiter='\t', header=None, 
                 names=['sentence_source', 'label', 'label_notes', 'sentence'])

"""
print(df.shape)
print(df.sample(10))
"""

# ---- Creating sentence, Label lists and adding Bert tokens ----
sentences = df.sentence.values
# Adding CLS and SEP tokens at the beginning and end of each sentence for BERT
sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
labels = df.label.values


# ---- Activating the BERT Tokenizer ----
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

print("Tokenuze the first sentence:")
print(tokenized_texts[0])

# ---- Processing the data ----
# Set the maximum sequence length. 
