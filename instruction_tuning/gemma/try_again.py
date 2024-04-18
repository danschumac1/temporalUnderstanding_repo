"""
Created on 04/09/2024

@author: Dan Schumacher
"""

import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, GenerationConfig
from tqdm import tqdm
from torchmetrics import Accuracy
from types import SimpleNamespace

# Set the working directory
os.chdir('/home/dan/TemporalUnderstandingInLLMs')

# Load JSONL data
def load_jsonl(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

train_ds_packed = load_jsonl('/home/dan/TemporalUnderstandingInLLMs/data/preprocessed/train/train_GEMMA_no_context_packed.jsonl')
eval_ds_packed = load_jsonl('/home/dan/TemporalUnderstandingInLLMs/data/preprocessed/dev/packed/dev_GEMMA_no_context_packed.jsonl')

# Data loader setup
batch_size = 8
train_dataloader = DataLoader(train_ds_packed, batch_size=batch_size, collate_fn=default_data_collator)
eval_dataloader = DataLoader(eval_ds_packed, batch_size=batch_size, collate_fn=default_data_collator, shuffle=False)

# Configuration settings
config = SimpleNamespace(
    model_id='google/gemma-2b-it',
    dataset_name="gemma_no_context",
    precision=torch.bfloat16,
    n_freeze=24,
    lr=2e-4,
    n_eval_samples=10,
    max_seq_len=1024,
    epochs=3,
    gradient_accumulation_steps=32 // batch_size,
    batch_size=batch_size,
    log_model=False,
    mom=0.9,
    gradient_checkpointing=True,
    freeze_embed=True,
    total_train_steps=None  # This will be calculated next
)

config.total_train_steps = config.epochs * len(train_dataloader) // config.gradient_accumulation_steps


# Model setup
model = AutoModelForCausalLM.from_pretrained(
    config.model_id,
    device_map=1,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    use_cache=False,
)

# Layer freezing
for param in model.parameters():
    param.requires_grad = False
for i, layer in enumerate(model.model.layers):
    if i >= config.n_freeze:
        for param in layer.parameters():
            param.requires_grad = True
if config.freeze_embed:
    model.model.embed_tokens.weight.requires_grad_(False)
if config.gradient_checkpointing:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# Optimizer and scheduler
optim = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.99), eps=1e-5)
scheduler = get_cosine_schedule_with_warmup(
    optim,
    num_training_steps=config.total_train_steps,
    num_warmup_steps=config.total_train_steps // 10,
)
# Loss function
def loss_fn(x, y):
    return torch.nn.functional.cross_entropy(x.view(-1, x.shape[-1]), y.view(-1))

# Utility to move tensors to GPU
def to_gpu(batch):
    thingy =  {k: v.to('cuda:1') for k, v in batch.items()}
    return thingy

def validate():
    model.eval()
    eval_acc = Accuracy()
    eval_loss = 0
    for batch in eval_dataloader:
        batch = to_gpu(batch)
        out = model(**batch)
        loss = loss_fn(out.logits, batch["labels"])
        eval_acc.update(out.logits, batch["labels"])
        eval_loss += loss.item()
    avg_loss = eval_loss / len(eval_dataloader)
    avg_acc = eval_acc.compute()
    print(f'Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_acc:.4f}')
    model.train()


# Training loop with autocast
from torch.cuda.amp import autocast  # Correct import for using autocast

model.train()
train_step = 0
pbar = tqdm(total=config.total_train_steps)

# for name, param in model.named_parameters():
#     print(name, param.requires_grad)
# print("Does loss require grad?", loss.requires_grad)
# print("Loss grad_fn:", loss.grad_fn)
# out = model(**batch)
# print("Does model output require grad?", out.logits.requires_grad)
# with torch.no_grad():
#     for param in model.parameters():
#         param.requires_grad = True


for epoch in range(config.epochs):
    for step, batch in enumerate(train_dataloader):
        batch = to_gpu(batch)
        with autocast():  # Use autocast with default settings to test
            out = model(**batch)
            loss = loss_fn(out.logits, batch["labels"]) / config.gradient_accumulation_steps
            loss.backward()
        if step % config.gradient_accumulation_steps == 0:
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)
            train_step += 1
            pbar.update(1)
    validate()


# If saving the model is desired
# model_path = "models/model_name.pt"
# torch.save(model.state_dict(), model_path)
