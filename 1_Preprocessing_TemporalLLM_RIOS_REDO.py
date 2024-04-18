#region # DATE AND AUTHOR
"""
Created on 04/08/2024

@author: Dan Schumacher
"""
#endregion
#region IMPORTS
# IMPORTS
# =============================================================================
import os
os.getcwd()
os.chdir('/home/dan/TemporalUnderstandingInLLMs')

import pandas as pd
from transformers import AutoTokenizer
import json
import wandb
import numpy as  np

#endregion
#region # DATA READING
# =============================================================================
# DATA READING
# =============================================================================
# LOAD IN TRAIN TEST AND SPLIT
# set file paths
# Shift the 'relevant_context' to create 'random_context'

train_path = './data/final/train.csv'


# pull it in 
train = pd.read_csv(train_path, index_col='Unnamed: 0')
train['random_context'] = train['relevant_context'].shift(-1)
train.loc[0, 'random_context'] = train['relevant_context'].iloc[-1]
train[['random_context','relevant_context']].head(4)
train['type'] = 'train'
train['answer']
data_dict = train.to_dict(orient='list')
data_dict['answer']

#endregion
#region # GEMMA PREPROCESSING
# =============================================================================
# GEMMA PREPROCESSING
# =============================================================================
# set the special tokens
user_SOS_GEMMA = '<start_of_turn>user'
model_SOS_GEMMA = '<start_of_turn>model'
EOS_GEMMA = '<end_of_turn>'

# NO CONTEXT
GEMMA_no_context = []
for q in data_dict['question'].values():
    example_GEMMA = f"{user_SOS_GEMMA}\n{q}{EOS_GEMMA}\n{model_SOS_GEMMA}"
    GEMMA_no_context.append(example_GEMMA)
data_dict['GEMMA_no_context'] = GEMMA_no_context

# =============================================================================
# TOKENIZATION
# =============================================================================
# GET HIDDEN TOKEN
with open('./data/token.txt', 'r') as file:
    # Read the entire content of the file into a single string
    token = file.read()

# GEMMA TOKENIZATION
# Use the content as a string
Gemma_model_id = 'google/gemma-2b-it'
Gemma_tokenizer = AutoTokenizer.from_pretrained(Gemma_model_id, use_auth_token=token)
# we are padding with "</s>"
Gemma_tokenizer.pad_token = Gemma_tokenizer.eos_token	

#endregion
#region # Create dataset to tokenize
# =============================================================================
# Create dataset to tokenize
# =============================================================================
data_dict["answer"]
dataset = [
    {'prompt': p, 'output': o, 'example': p + o} for p, o in zip(
        data_dict['GEMMA_no_context'],
        data_dict['answer']
        )
    ]
data_dict['answer'][0]
#endregion
#region # PACKING
# =============================================================================
# PACKING
# =============================================================================
# @$@ Can I get away with more?  
max_seq_length = 1024

def pack(dataset, max_seq_len=1024):
    # Tokenize the samples in the dataset using the tokenizer
    tkds_ids = Gemma_tokenizer([s["example"] for s in dataset])["input_ids"]
    
    # Create a list to store all token ids from all examples
    all_token_ids = []
    for tokenized_input in tkds_ids:
        # Extend the list with token ids of each example and add EOS token id
        all_token_ids.extend(tokenized_input + [Gemma_tokenizer.eos_token_id])
    
    # Initialize a list to store packed datasets
    packed_ds = []
    # Iterate over all token ids with a step of max_seq_len + 1
    for i in range(0, len(all_token_ids), max_seq_len+1):
        # Extract input_ids and labels for each sequence
        input_ids = all_token_ids[i : i + max_seq_len+1]
        # Check if the sequence length matches max_seq_len + 1
        if len(input_ids) == (max_seq_len+1):
            # Append the input_ids and labels as a dictionary to packed_ds
            packed_ds.append({"input_ids": input_ids[:-1], "labels": input_ids[1:]})  # Shift input_ids to create labels
            # Note: If you use the model.output.loss, you don't need to shift; it's done for you.
    # Return the packed dataset
    return packed_ds

train_ds_packed = pack(dataset)

#endregion
#region # STORING PREPROCESSED DATASETS ON W&B
# =============================================================================
# STORING PREPROCESSED DATASETS ON W&B
# =============================================================================
def save_jsonl(data, filename):
    with open(filename, 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')

# dump everything to jsonl files
save_jsonl(train_ds_packed, "./data/train_GEMMA_no_context_packed.jsonl")
