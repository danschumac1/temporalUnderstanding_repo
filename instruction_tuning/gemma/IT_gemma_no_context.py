"""
Created on 04/09/2024

@author: Dan Schumacher
"""

#region # IMPORTS
# =============================================================================
# IMPORTS
# =============================================================================
import os
os.chdir('/home/dan/TemporalUnderstandingInLLMs')
# import wandb
# from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator, AutoModelForCausalLM
# from datasets import load_from_disk
# import accelerate
from types import SimpleNamespace
# import openai
from transformers import GenerationConfig
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from torchmetrics import Accuracy

#endregion
#region # DATA LOADING
# =============================================================================
# DATA LOADING
# =============================================================================
def load_jsonl(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

train_ds_packed = load_jsonl('/home/dan/TemporalUnderstandingInLLMs/data/preprocessed/train/train_GEMMA_no_context_packed.jsonl')
eval_ds_packed = load_jsonl('/home/dan/TemporalUnderstandingInLLMs/data/preprocessed/dev/packed/dev_GEMMA_no_context_packed.jsonl')
max_seq_len = 1024

#endregion
#region # DATA LOADER 
# =============================================================================
# DATA LOADER
# =============================================================================

batch_size = 4  # I have an A100 GPU with 40GB of RAM 😎


train_dataloader = DataLoader(
    train_ds_packed,
    batch_size=batch_size,
    collate_fn=default_data_collator, # we don't need any special collator 😎
)

eval_dataloader = DataLoader(
    eval_ds_packed,
    batch_size=batch_size,
    collate_fn=default_data_collator,
    shuffle=False,
)

#  CHECK TO SEE WHAT THE BATCH LOOKS LIKE
b = next(iter(train_dataloader))
b.keys(), b['input_ids'][0][:25], b['labels'][0][:25]
b['input_ids'].shape
b['labels'].shape

#endregion
#region # TRAINING LOOP
# =============================================================================
# TRAINING LOOP
# =============================================================================

gradient_accumulation_steps = 32 // batch_size


config = SimpleNamespace(
    model_id='google/gemma-2b-it',
    dataset_name= "gemma_no_context",
    precision="bf16",               # faster and better than fp16, requires new GPUs
    n_freeze=24,                    # How many layers we don't train, LLama 7B has 32.
    lr=2e-4,                        # the learning rate
    n_eval_samples=10,              # How many samples to generate on validation
    max_seq_len=max_seq_len,        # Length of the sequences to pack
    epochs=3,                       # we do 3 pasess over the dataset.
    gradient_accumulation_steps=gradient_accumulation_steps,  # evey how many iterations we update the gradients, simulates larger batch sizes
    batch_size=batch_size,          # what my GPU can handle, depends on how many layers are we training  
    log_model=False,                # upload the model to W&B?
    mom=0.9,                        # optim param
    gradient_checkpointing = True,  # saves even more memory
    freeze_embed = True,            # why train this? let's keep them frozen ❄️
)


config.total_train_steps = config.epochs * len(train_dataloader) // config.gradient_accumulation_steps

with open('./data/token.txt', 'r') as file:
    # Read the entire content of the file into a single string
    token = file.read()

model = AutoModelForCausalLM.from_pretrained(
    config.model_id,
    device_map=1, # Which GPU? 0 or 1?
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    use_cache=False,
    token = token,
)

#endregion
#region # FREEZING
# =============================================================================
# FREEZING
# =============================================================================
n_freeze = 24. # you can play with this parameter

# freeze layers (disable gradients)
# Freeze all parameters initially
for param in model.parameters(): 
    param.requires_grad = False

# Unfreeze lm_head parameters
for param in model.lm_head.parameters(): 
    param.requires_grad = True

# Freeze all parameters initially
for i, layer in enumerate(model.model.layers):
    if i >= n_freeze:
        for param in layer.parameters():
            param.requires_grad = True

# Freeze the Embeddings
if config.freeze_embed:
    model.model.embed_tokens.weight.requires_grad_(False)

# Use Gradent checkpoints to save EVEN MORE MEMORY
if config.gradient_checkpointing:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

#endregion
#region # OPTIMIZER AND SCHEDULER
# =============================================================================
# OPTIMIZER AND SCHEDULER
# =============================================================================
optim = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9,0.99), eps=1e-5)
scheduler = get_cosine_schedule_with_warmup(
    optim,
    num_training_steps=config.total_train_steps,
    num_warmup_steps=config.total_train_steps // 10,
)

def loss_fn(x, y):
    "A Flat CrossEntropy" 
    return torch.nn.functional.cross_entropy(x.view(-1, x.shape[-1]), y.view(-1))

#endregion
#region # SAMPLING FROM THE MODEL
# =============================================================================
# SAMPLING FROM THE MODEL
# =============================================================================
gen_config = GenerationConfig.from_pretrained(config.model_id)

# create simple sample function
# see what the model is outputting
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b-it', token=token)
tokenizer.pad_token = tokenizer.eos_token	

def generate(prompt, max_new_tokens=100, gen_config=gen_config):
    with torch.inference_mode():
        tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        output = model.generate(tokenized_prompt, 
                            max_new_tokens=max_new_tokens, 
                            generation_config=gen_config)
    return tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)

# Define a configuration for testing
test_config = SimpleNamespace(
    max_new_tokens=100,  # Adjust according to your model's capabilities
    gen_config=SimpleNamespace(temperature=0.9, top_p=0.95)  # Example values
)

# def prompt_table(prompts, max_new_tokens, temperature, top_p, log=True):
#     table = wandb.Table(columns=["prompt", "generation", "concat", "max_new_tokens", "temperature", "top_p"])
#     for prompt in tqdm(prompts, desc="Generating prompts"):
#         out = generate(prompt, max_new_tokens, SimpleNamespace(temperature=temperature, top_p=top_p))
#         table.add_data(prompt, out, prompt + out, max_new_tokens, temperature, top_p)
#     if log:
#         wandb.log({"predictions": table})
#     return table


# def prompt_table(prompts, log=True):
#     table = wandb.Table(columns=["prompt", "generation", "concat", "max_new_tokens", "temperature", "top_p"])
#     for prompt in progress_bar(prompts):
#         out = generate(prompt, test_config.max_new_tokens, test_config.gen_config)
#         table.add_data(prompt, out, prompt+out, test_config.max_new_tokens, test_config.gen_config.temperature, test_config.gen_config.top_p)
#     if log:
#         wandb.log({"predictions":table})
#     return table

#endregion
#region # VALIDATION STEP
# =============================================================================
# VALIDATION STEP
# =============================================================================
# @$@ to_gpu is undefined
# @$@ Accuracy is undefined
# @$@ Is accuracy even the metric that we want to be using? 

@torch.no_grad()
def validate():
    model.eval();
    # eval_acc = Accuracy()

    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = to_gpu(batch)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
            loss = loss_fn(out.logits, batch["labels"])  # you could use out.loss and not shift the dataset
        # eval_acc.update(out.logits, batch["labels"])
    # we log results at the end
    # wandb.log({"eval_loss": loss.item(),
    #            "eval_accuracy": eval_acc.compute()})
    # prompt_table(eval_dataset[:config.n_eval_samples], log=True)
    model.train();

#endregion
#region # PYTORCH TRAINING LOOP FOR LLM
# =============================================================================
# PYTORCH TRAINING LOOP FOR LLM
# =============================================================================
# Training

def to_gpu(batch):
    # Assuming your batch is a dictionary of tensors
    thingy = {k: v.to('cuda:1') for k, v in batch.items()}
    return thingy

# Example to ensure the model's output layers require gradients
for param in model.lm_head.parameters():
    param.requires_grad = True  # Make sure the output layer is trainable


model.train()
train_step = 0
pbar = tqdm(total=config.total_train_steps)
for epoch in range(config.epochs):
    for step, batch in enumerate(train_dataloader):
        batch = to_gpu(batch)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
            loss = loss_fn(out.logits, batch["labels"]) / config.gradient_accumulation_steps  # you could use out.loss and not shift the dataset  
            loss.backward()
        if step%config.gradient_accumulation_steps == 0:
            # # we can log the metrics to W&B
            # wandb.log({"train/loss": loss.item() * config.gradient_accumulation_steps,
            #            "train/accuracy": acc.update(out.logits, batch["labels"]),
            #            "train/learning_rate": scheduler.get_last_lr()[0],
            #            "train/global_step": train_step})
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)
            train_step += 1
            pbar.update(1)
    validate()
pbar.close() # 240562 # 360773




# Assuming acc is a predefined accuracy metric
# acc = Accuracy()
model.train()
train_step = 0
pbar = tqdm(total=config.total_train_steps)

for epoch in range(config.epochs):
    for step, batch in enumerate(train_dataloader):
        batch = to_gpu(batch)  # Ensure your batch is properly moved to GPU
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
            loss = loss_fn(out.logits, batch["labels"]) / config.gradient_accumulation_steps
            loss.backward()
        if step % config.gradient_accumulation_steps == 0:
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)

            # Logging the metrics without W&B
            current_loss = loss.item() * config.gradient_accumulation_steps
            # current_acc = acc.update(out.logits, batch["labels"])
            current_lr = scheduler.get_last_lr()[0]

            print(f"Epoch: {epoch}, Step: {train_step}, Loss: {current_loss:.4f}")
            # , Accuracy: {current_acc:.4f}, Learning Rate: {current_lr:.6f}")

            train_step += 1
            pbar.update(1)

    # Perform validation and print validation metrics
    validate()

pbar.close()

# @torch.no_grad()
def validate():
    model.eval()
    eval_loss = 0
    # eval_acc = Accuracy()  # Ensure Accuracy is a suitable metric class # KILL IT WITH FIRE
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = to_gpu(batch)
            out = model(**batch)
            loss = loss_fn(out.logits, batch["labels"]) 
            # eval_acc.update(out.logits, batch["labels"])   # KILL IT WITH FIRE
            eval_loss += loss.item()

    avg_loss = eval_loss / len(eval_dataloader)
    # avg_acc = eval_acc.compute() # KILL IT WITH FIRE

    print(f"Validation Loss: {avg_loss:.4f}")
    # , Validation Accuracy: {avg_acc:.4f}") # KILL IT WITH FIRE
    model.train()



# save the file. 
# we save the model checkpoint at the end


# Model name (gemma or llama)
model_used = 'gemma'
# Model Subfolder (gemma or llama)
subfold = 'gemma'
# context (no, rel, random, or wd)
context = 'no'

file_path = f'/home/dan/TemporalUnderstandingInLLMs/models/{subfold}/{model_used}_{context}_context.pt'
file_path

torch.save(model, file_path)

#endregion

# END OF THIS py


# TO RUN VIA LINUX
# nohup python IT_gemma_no_context.py > /home/dan/TemporalUnderstandingInLLMs/model_model_outputs/gemma_no_training.out &


#endregion
#region # NOTES
# =============================================================================
# NOTES
# =============================================================================


# in a new .py
# load saved model
# load the generate function
    # GO back and make sure answer aren't in the unpacked.
# generate output for each test data
# ^^^ give to exact, contains, bem evaluation models.