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
dev_path = './data/final/dev.csv'
test_path = './data/final/test.csv'

# pull it in 
train = pd.read_csv(train_path, index_col='Unnamed: 0')
train['random_context'] = train['relevant_context'].shift(-1)
train.loc[0, 'random_context'] = train['relevant_context'].iloc[-1]


dev = pd.read_csv(dev_path, index_col='Unnamed: 0')
dev['random_context'] = dev['relevant_context'].shift(-1)
dev.loc[0, 'random_context'] = dev['relevant_context'].iloc[-1]


test = pd.read_csv(test_path, index_col='Unnamed: 0')
test['random_context'] = test['relevant_context'].shift(-1)
test.loc[0, 'random_context'] = test['relevant_context'].iloc[-1]

# create a dataframe of them all...
all_data = pd.concat([train, dev, test])
# convert to a dictionary for manipulation
data_dict = all_data.to_dict()

# LOGGING
with wandb.init(project="temporalUnderstanding") as run:
    at = wandb.Artifact(
        name="TemporalUnderstandingInLLMs",
        type="dataset",
        description="Temporal Questions Dataset",
        metadata={
            "Original Source 1": "https://github.com/wenhuchen/Time-Sensitive-QA/tree/main/dataset",
            "Original Source 2": "https://www.dropbox.com/sh/fdepuisdce268za/AACtiPDaO_RwLCwhIwaET4Iba?dl=0"
        },
    )

    at.add_file(train_path)
    at.add_file(dev_path)
    at.add_file(test_path)
    
    # Prepare the columns from the keys of data_dict
    columns = list(data_dict.keys())
    
    # Create a W&B Table with these columns
    table = wandb.Table(columns=columns)
    
    # Assuming all lists in data_dict are of the same length
    # Use zip to iterate over the values of each list in parallel
    for row in zip(*data_dict.values()):
        # Add the row to the table
        table.add_data(*row)
    
    # Add the table to the artifact
    at.add(table, name="Dataset Table")
    
    # Log the artifact to W&B
    run.log_artifact(at)

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

# RELEVANT CONTEXT
GEMMA_no_context = []
for c,q in zip(data_dict['relevant_context'].values(), data_dict['question'].values()):
    example_GEMMA = f"{user_SOS_GEMMA}\n{c}\n\n{q}{EOS_GEMMA}\n{model_SOS_GEMMA}"
    GEMMA_no_context.append(example_GEMMA)

# WRONG DATE CONTEXT
GEMMA_wrong_date_context = []
for c,q in zip(data_dict['wrong_date_context'].values(), data_dict['question'].values()):
    example_GEMMA = f"{user_SOS_GEMMA}\n{c}\n\n{q}{EOS_GEMMA}\n{model_SOS_GEMMA}"
    GEMMA_wrong_date_context.append(example_GEMMA)

# RANDOM CONTEXT
GEMMA_random_context = []
for c,q in zip(data_dict['random_context'].values(), data_dict['question'].values()):
    example_GEMMA = f"{user_SOS_GEMMA}\n{c}\n\n{q}{EOS_GEMMA}\n{model_SOS_GEMMA}"
    GEMMA_random_context.append(example_GEMMA)


#endregion
#region # LLAMA-2 PREPROCESSING
# =============================================================================
# LLAMA-2 PREPROCESSING 
# =============================================================================
# Define unique tokens for Llama 2 interactions
user_SOS_LLAMA = '<s>[INST]'  # Start of an instruction segment
model_SOS_LLAMA = '</s><s>[INST]'  # End and start a new instruction segment
EOS_LLAMA = '[/INST]</s>'  # End of an instruction segment

# Define the system prompt that guides the model's behavior
system_prompt = """<<SYS>> You are a helpful, respectful, and honest assistant. Answer the questions given the context. <</SYS>>"""

# NO CONTEXT PREPROCESSING
llama_no_context = [
    f"{user_SOS_LLAMA} {system_prompt}\n{q}{EOS_LLAMA}" 
    for q in data_dict['question'].values()
]

# RELEVANT CONTEXT PREPROCESSING
llama_relevant_context = [
    f"{user_SOS_LLAMA} {system_prompt}\n{c}\n\n{q}{EOS_LLAMA}" 
    for c, q in zip(data_dict['relevant_context'].values(), data_dict['question'].values())
]

# WRONG DATE CONTEXT
llama_wrong_date_context = [
    f"{user_SOS_LLAMA} {system_prompt}\n{c}\n\n{q}{EOS_LLAMA}" 
    for c, q in zip(data_dict['wrong_date_context'].values(), data_dict['question'].values())
]

# RANDOM CONTEXT
llama_random_context = [
    f"{user_SOS_LLAMA} {system_prompt}\n{c}\n\n{q}{EOS_LLAMA}" 
    for c, q in zip(data_dict['random_context'].values(), data_dict['question'].values())
]

# Example to verify the format
#endregion
#region # TOKENIZATION
# =============================================================================
# TOKENIZATION
# =============================================================================
# GET HIDDEN TOKEN
with open('./data/token.txt', 'r') as file:
    # Read the entire content of the file into a single string
    token = file.read()

# GEMMA TOKENIZATION
# Use the content as a string
Gemma_model_id = 'google/gemma-7b'
Gemma_tokenizer = AutoTokenizer.from_pretrained(Gemma_model_id, use_auth_token=token)
# we are padding with "</s>"
Gemma_tokenizer.pad_token = Gemma_tokenizer.eos_token

# LLAMA-2 TOKENIZATION
llama_model_id = '/home/anrios/llama2/7b-chat'
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
# we are padding with "</s>"
llama_tokenizer.pad_token = llama_tokenizer.eos_token	

#endregion
#region # TRAIN TEST SPLIT
# =============================================================================
# TRAIN TEST SPLIT 
# =============================================================================
# I need to convert my data_dict obj into a list
# With relevant context only

dataset = [
    {'prompt': p, 'output': o, 'example': p + o} for p, o in zip(
        data_dict['no_context_prompt'],
        data_dict['Answer']
        )
    ]

import random
random.seed(27)
random.shuffle(dataset)

# our dataset has 500 examples
# train on everything but the last 100 rows
train_dataset = dataset[:-100]
# evaluate on the last 100 rows
eval_dataset = dataset[-100:]

# push to W&B
train_table = wandb.Table(dataframe=pd.DataFrame(train_dataset))
eval_table  = wandb.Table(dataframe=pd.DataFrame(eval_dataset))
with wandb.init(project="temporalUnderstanding", job_type="split_data"):
    wandb.log({"train_dataset":train_table, "eval_dataset":eval_table})

#endregion
#region # PACKING
# =============================================================================
# PACKING
# =============================================================================

# @$@ Can I get away with more?  
max_seq_length = 1024

def pack(dataset, max_seq_len=1024):
    # Tokenize the samples in the dataset using the tokenizer
    tkds_ids = tokenizer([s["example"] for s in dataset])["input_ids"]
    
    # Create a list to store all token ids from all examples
    all_token_ids = []
    for tokenized_input in tkds_ids:
        # Extend the list with token ids of each example and add EOS token id
        all_token_ids.extend(tokenized_input + [tokenizer.eos_token_id])
    
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

train_ds_packed = pack(train_dataset)
eval_ds_packed = pack(eval_dataset)

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
save_jsonl(train_ds_packed, "./data/train_packed_TQ.jsonl")
save_jsonl(eval_ds_packed, "./data/eval_packed_TQ.jsonl")


# Create a W&B artifact
packed_at = wandb.Artifact(
    name="packed_TQ",
    type="dataset",
    description="TQ dataset packed in sequences",
    metadata={"max_seq_len":1024, "model_id":model_id})

packed_at.add_file("./data/train_packed_TQ.jsonl") # called wrong name
packed_at.add_file("./data/eval_packed_TQ.jsonl") # called wrong name

# log the artifact to the project, we can give this run a job_type like `preprocess`
with wandb.init(project="temporalUnderstanding", job_type="preprocess"):
    wandb.log_artifact(packed_at)
