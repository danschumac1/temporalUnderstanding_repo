
#region # IMPORTS 
# =============================================================================
# IMPORTS
# =============================================================================

import trl
# from trl.train import SFTTrainer, TrainingArguments
import pandas as pd

from datasets import load_dataset, Dataset, load_metric
import pandas as pd
import evaluate
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    AutoModelForSequenceClassification,
    Trainer,
    TrainerCallback
)
from copy import deepcopy
import csv

# SET DIRECTORY
import os
os.getcwd()
os.chdir('./DeepLearning/Temporal')

#endregion
#region # LOAD DATA
# =============================================================================
# LOAD DATA
# =============================================================================
# READ IN THE DATA
df = pd.read_csv('./data/TQ_revised.csv')

# RELEVANT CONTEXT, QUESTON
df['RCQ'] = [f'Context: {RC}\n\nQuestion: {Q}\n\n' for RC,Q in zip(df['relevant_context'],df['Question'])]

# WRONG DATE CONTEXT, QUESTION
df['WD_RCQ'] = [f'Context: {WDC}\n\nQuestion: {Q}\n\n' for WDC,Q in zip(df['wrong_date_context'],df['Question'])]

# IRRELEVANT CONTEXT
# grab the context from the next row
df['IC'] = df['relevant_context'].shift(1)
# put the last context as the first row
df['IC'].iloc[0] = df['relevant_context'].iloc[-1]

# IRRELEVANT AND WRONG DATE CONTEXT
# grab the context from the next row
df['WD_ICQ'] = df['wrong_date_context'].shift(1)
# put the last context as the first row
df['WD_ICQ'].iloc[0] = df['wrong_date_context'].iloc[-1]

# make an idx variable
df.reset_index(inplace=True, names=['idx'])

# CAST EVERYTHING INTO DATASET OBJECTS
df_RCQ = Dataset.from_pandas(df[['idx','RCQ','Answer']])
# rename to CQ to standardize across different groups
df_RCQ = df_RCQ.rename_column('RCQ','CQ',)

df_WD_RCQ = Dataset.from_pandas(df[['idx','WD_RCQ','Answer']])
df_WD_RCQ = df_WD_RCQ.rename_column('WD_RCQ','CQ')

df_IQ = Dataset.from_pandas(df[['idx','IC','Answer']])
df_IQ = df_IQ.rename_column('IC','CQ')

df_WD_ICQ = Dataset.from_pandas(df[['idx','WD_ICQ','Answer']])
df_WD_ICQ = df_WD_ICQ.rename_column('WD_ICQ','CQ')


# for now... 
# later we will need a dev and test? but it is just so small :)
train = df_RCQ


#endregion
#region # TOKENIZATION
# =============================================================================
# Tokenize the Data
# =============================================================================

checkpoint = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    truncation = True,
    padding = True
    )

def cqa_tokenize_function(batch):
    return tokenizer(
        batch['CQ'],
        truncation=True,
        padding='max_length',
        max_length=512
    )

tokenized_train = train.map(cqa_tokenize_function, batched=True)
tokenized_train = tokenized_train.remove_columns(['CQ'])


#endregion
#region # FOLLOW ALONG WITH WEBSITE
# =============================================================================
# FOLLOW ALONG WITH WEBSITE
# =============================================================================
import trl
# from trl.train import SFTTrainer, TrainingArguments


os.environ["WANDB_PROJECT"] = "alpaca_ft"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


training_args = TrainingArguments(
    report_to="wandb", # enables logging to W&B 😎
    per_device_train_batch_size=16,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    gradient_accumulation_steps=2, # simulate larger batch sizes
)


trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    packing=True, # pack samples together for efficient training
    max_seq_length=1024, # maximum packed length 
    args=training_args,
    formatting_func=formatting_func, # format samples with a model schema
)
trainer.train()



# tokenized_dev = dev.map(cqa_tokenize_function, batched=True)
# tokenized_dev = tokenized_dev.remove_columns(['CQ'])


# training_args = TrainingArguments(
#     report_to="wandb", # enables logging to W&B 😎
#     per_device_train_batch_size=16,
#     learning_rate=2e-4,
#     lr_scheduler_type="cosine",
#     num_train_epochs=3,
#     gradient_accumulation_steps=2, # simulate larger batch sizes
# )


# trainer = SFTTrainer(
#     model,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     packing=True, # pack samples together for efficient training
#     max_seq_length=1024, # maximum packed length 
#     args=training_args,
#     formatting_func=formatting_func, # format samples with a model schema
# )

# trainer.train()
