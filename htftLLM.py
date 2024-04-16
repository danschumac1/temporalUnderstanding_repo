#region
# =============================================================================
# IMPORTS
# =============================================================================

import json
import wandb
import torch


from transformers import (
    AutoTokenizer,
    # DataCollatorWithPadding,
    # TrainingArguments,
    # AutoModelForSequenceClassification,
    # Trainer,
    # TrainerCallback,
    # DebertaV2Tokenizer

)

import os
os.getcwd()
os.chdir('./Learning')

#endregion
#region # LOAD DATA
# =============================================================================
# LOAD DATA
# =============================================================================

with open("./data/alpaca_data.json", "r") as f:
    alpaca = json.load(f)


with wandb.init(project="alpaca_ft"):
    at = wandb.Artifact(
        name="alpaca_gpt4", 
        type="dataset",
        description="A GPT4 generated Alpaca like dataset for instruction finetunning",
        metadata={"url":"https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM#how-good-is-the-data"},
    )
    at.add_file("./data/alpaca_data.json")
    
    # table
    table = wandb.Table(columns=list(alpaca[0].keys()))
    for row in alpaca:
        table.add_data(*row.values())

len(alpaca)
type(alpaca)
one_row = alpaca[232]
for obj in alpaca[0]:
    print(obj)
one_row

#endregion
#region # PREPROCESSING
# =============================================================================
# PREPROCESSING
# =============================================================================
def prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n").format_map(row)


def prompt_input(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n").format_map(row)

# merge no_inputs and inputs into a single function
def create_prompt(row):
    return prompt_no_input(row) if row["input"] == "" else prompt_input(row)


prompts = [create_prompt(row) for row in alpaca]  # all LLM inputs are here

# end of string tokens
EOS_TOKEN = "</s>"
outputs = [row['output'] + EOS_TOKEN for row in alpaca]

# store the stuff
dataset = [{"prompt":s, "output":t, "example": s+t} for s, t in zip(prompts, outputs)]
# it is possible to store preprocessed dataset as a W&B Artifact to avoid re doing this ^

#endregion
#region # TOKENIZATION
# =============================================================================
# TOKENIZATION 
# =============================================================================
from transformers import AutoModelForCausalLM, AutoConfig

llama_config = AutoConfig.from_pretrained("./llama-2-7b-chat")
llama_model = AutoModelForCausalLM.from_config(llama_config)

# for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llama_model.to(device)

model_id = 'meta-llama/Llama-2-7b-hf'

tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token

