"""
Created on 04/08/2024

@author: Dan Schumacher
"""
import os
os.getcwd()
os.chdir('/home/dan/TemporalUnderstandingInLLMs')
#endregion
#region # IMPORTS
# IMPORTS
# =============================================================================
import pandas as pd
from transformers import AutoTokenizer
import json
import wandb

#endregion
#region # DATA READING
# =============================================================================
# DATA READING
# =============================================================================
dataset_file_path = './data/TQ_revised.csv'
df = pd.read_csv(dataset_file_path)
data_dict = df.to_dict()

new_data_dict = {}
for key in data_dict.keys():
    new_data_dict[key] = [s for s in data_dict[key].values()];

df = pd.DataFrame(new_data_dict)
df['random_context'] = df['relevant_context'].shift(-1)
df.loc[df.index[-1], 'random_context'] = df['relevant_context'].iloc[0]

data_dict = new_data_dict

with wandb.init(project="temporalUnderstanding"):
    at = wandb.Artifact(
        name="TemporalUnderstandingInLLMs",
        type="dataset",
        description="Temporal Questions Dataset",
        metadata={"Original Source": "@$@ put URL HERE"},  # Ensure you replace @$@ with the actual URL
    )
    at.add_file("./data/TQ_revised.csv")
    
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
    at.add(table, "Dataset Table")
    
    # Log the artifact to W&B
    wandb.log_artifact(at)

#endregion
#region # DATA PREPROCESSING
# =============================================================================
# DATA PREPROCESSING
# =============================================================================

# This will always be the preamble
prompt_head = '''Below is a Question that needs to be answered. Write a response that answers the question to the best of your abilities.
'''

# NO CONTEXT
data_dict['no_context_prompt'] = [f'{prompt_head}\n### Question:\n{Q}\n\n### Response:\n' for Q in data_dict['Question']]
# CORRECT CONTEXT
data_dict['rel_context_prompt'] = [f'{prompt_head}\n### Question:\n{Q}\n\n### Response:\n' for Q in zip(data_dict['Question'], data_dict['relevant_context'])]
# MIS-DATED CONTEXT
data_dict['wrongDate_context_prompt'] = [f'{prompt_head}\n### Question:\n{Q}\n\n### Response:\n' for Q in zip(data_dict['Question'], data_dict['wrong_date_context'])]

#endregion
#region # GEMMA PREPROCESSING
# =============================================================================
# GEMMA PREPROCESSING
# =============================================================================
user_SOS_GEMMA = '<start_of_turn>user'
model_SOS_GEMMA = '<start_of_turn>model'
EOS_GEMMA = '<end_of_turn>'

print('<start of turn>user\nPROMPTHEAD\nCONTEXT BLAH BLAH BLAH\nQUESTION: who is superman? <end_of_turn>\n<start_of_turn>model')

data_dict.keys()

# NO CONTEXT
GEMMA_no_context = []
for q in data_dict['Question']:
    example_GEMMA = f"{user_SOS_GEMMA}\n{q}{EOS_GEMMA}\n{model_SOS_GEMMA}"
    GEMMA_no_context.append(example_GEMMA)

# RELEVANT CONTEXT
GEMMA_no_context = []
for c,q in zip(data_dict['relevant_context'], data_dict['Question']):
    example_GEMMA = f"{user_SOS_GEMMA}\n{c}\n\n{q}{EOS_GEMMA}\n{model_SOS_GEMMA}"
    GEMMA_no_context.append(example_GEMMA)

# WRONG DATE CONTEXT
GEMMA_wrong_date_context = []
for c,q in zip(data_dict['wrongDate_context_prompt'], data_dict['Question']):
    example_GEMMA = f"{user_SOS_GEMMA}\n{c}\n\n{q}{EOS_GEMMA}\n{model_SOS_GEMMA}"
    GEMMA_wrong_date_context.append(example_GEMMA)

print(example_GEMMA)
print("<start_of_turn>user Which two U.S. States had a border dispute that had to be settled by the U.S. Supreme Court in April of 2001?<end_of_turn>\n<start_of_turn>model\n")

#endregion
#region # LLAMA2 PREPROCESSING
# =============================================================================
# LLAMA2 PREPROCESSING
# =============================================================================
EOS_TOKEN_LLAMA2 = "</s>"



# EOS tokens for answers?
EOS_TOKEN = "</s>"
for i in range(len(data_dict['Answer'])):
    if data_dict['Answer'][i].endswith(EOS_TOKEN):
        pass
    else:
        data_dict['Answer'][i] = data_dict['Answer'][i] + EOS_TOKEN
#endregion
#region # TOKENIZATION
# =============================================================================
# TOKENIZATION
# =============================================================================
# this is the model we are using for tokenization
model_id = '/home/anrios/llama2/7b-chat'
tokenizer = AutoTokenizer.from_pretrained(model_id)

# we are padding with "</s>"
tokenizer.pad_token = tokenizer.eos_token	

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
