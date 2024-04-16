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
import wandb
from pathlib import Path
import json
from torch.utils.data import DataLoader
from transformers import default_data_collator, AutoModelForCausalLM
from datasets import load_from_disk
import torch
import accelerate
from types import SimpleNamespace
import openai
from transformers import GenerationConfig
from transformers import get_cosine_schedule_with_warmup

#endregion
#region # DATA LOADING
# =============================================================================
# DATA LOADING
# =============================================================================
run = wandb.init()
artifact = run.use_artifact('danschumac1-nlp/temporalUnderstanding/packed_TQ:v1', type='dataset')
artifact_dir = Path(artifact.download())

def load_jsonl(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

train_ds_packed = load_jsonl(artifact_dir/"train_packed_TQ.jsonl")
eval_ds_packed = load_jsonl(artifact_dir/"eval_packed_TQ.jsonl")
max_seq_len = artifact.metadata["max_seq_len"]

#endregion
#region # DATA LOADER 
# =============================================================================
# DATA LOADER
# =============================================================================

batch_size = 8  # I have an A100 GPU with 40GB of RAM 😎


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

#endregion
#region # TRAINING LOOP
# =============================================================================
# TRAINING LOOP
# =============================================================================

gradient_accumulation_steps = 32 // batch_size


config = SimpleNamespace(
    model_id='/home/anrios/llama2/7b-chat',
    dataset_name="alpaca-gpt4",
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


model = AutoModelForCausalLM.from_pretrained(
    config.model_id,
    device_map=0, # Which GPU? 0 or 1?
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    use_cache=False,
)

# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.version.cuda)

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

# create simple sample function
# see what the model is outputting
gen_config = GenerationConfig.from_pretrained(config.model_id)


def generate(prompt, max_new_tokens=100, gen_config=gen_config):
    with torch.inference_mode():
        tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        output = model.generate(tokenized_prompt, 
                            max_new_tokens=max_new_tokens, 
                            generation_config=gen_config)
    return tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)


def prompt_table(prompts, log=True):
    table = wandb.Table(columns=["prompt", "generation", "concat", "max_new_tokens", "temperature", "top_p"])
    for prompt in progress_bar(prompts):
        out = generate(prompt, test_config.max_new_tokens, test_config.gen_config)
        table.add_data(prompt, out, prompt+out, test_config.max_new_tokens, test_config.gen_config.temperature, test_config.gen_config.top_p)
    if log:
        wandb.log({"predictions":table})
    return table

#endregion
#region # VALIDATION STEP
# =============================================================================
# VALIDATION STEP
# =============================================================================
@torch.no_grad()
def validate():
    model.eval();
    eval_acc = Accuracy()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = to_gpu(batch)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
            loss = loss_fn(out.logits, batch["labels"])  # you could use out.loss and not shift the dataset
        eval_acc.update(out.logits, batch["labels"])
    # we log results at the end
    wandb.log({"eval_loss": loss.item(),
               "eval_accuracy": eval_acc.compute()})
    prompt_table(eval_dataset[:config.n_eval_samples], log=True)
    model.train();


#endregion
#region # PYTORCH TRAINING LOOP FOR LLM
# =============================================================================
# PYTORCH TRAINING LOOP FOR LLM
# =============================================================================
# TAKES ABOUT 2 HOURS
# wandb.init(project="alpaca_ft", # the project I am working on
#            tags=["baseline","7b"],
#            job_type="train",
#            config=config) # the Hyperparameters I want to keep track of


# # Training
# acc = Accuracy()
# model.train()
# train_step = 0
# pbar = tqdm(total=config.total_train_steps)
# for epoch in range(config.epochs):
#     for step, batch in enumerate(train_dataloader):
#         batch = to_gpu(batch)
#         with torch.amp.autocast("cuda", dtype=torch.bfloat16):
#             out = model(**batch)
#             loss = loss_fn(out.logits, batch["labels"]) / config.gradient_accumulation_steps  # you could use out.loss and not shift the dataset  
#             loss.backward()
#         if step%config.gradient_accumulation_steps == 0:
#             # we can log the metrics to W&B
#             wandb.log({"train/loss": loss.item() * config.gradient_accumulation_steps,
#                        "train/accuracy": acc.update(out.logits, batch["labels"]),
#                        "train/learning_rate": scheduler.get_last_lr()[0],
#                        "train/global_step": train_step})
#             optim.step()
#             scheduler.step()
#             optim.zero_grad(set_to_none=True)
#             train_step += 1
#             pbar.update(1)
#     validate()
# pbar.close()
# # we save the model checkpoint at the end
# save_model(
# 	model, 
# 	model_name=config.model_id.replace("/", "_"), 
# 	models_folder="models/", log=config.log_model)
    
# wandb.finish()

#endregion
#region # GPT-4 BASED EVALUATION
# =============================================================================
# GPT-4 BASED EVALUATION
# =============================================================================
def gpt4_judge(instruction, gen1, gen2, model="gpt-4"):
    system_prompt = ("You will be presented with a choice of two possible responses for an instruction"
                     "You have to pick the best one and give a reason why.\n"
                     "The reponse should follow the instructions and use the provided context if there is some\n"
                    "If both answers are equivalent, pick the value 0")
    message = "{instruction}\n Answer 1: \n{gen1}\n Answer 2:\n{gen2}".format(instruction=instruction, gen1=gen1, gen2=gen2)
    completion = openai.chat.completions.create(
        model=model,
        messages=[{"role": "system",
                   "content": system_prompt,
                  },
                  {"role": "user",
                   "content": message,
                  },],
        function_call = {"name": "make_choice"},
        functions = [{
                "name": "make_choice",
                "description": "Select the best generation and explain why",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "choice": {
                            "type": "integer",
                            "description": "the choosen alternative, zero if equivalent",
                        },
                        "argument":{
                            "type": "string",
                            "description": "Reason why the choice was made",},},},
                    "required": ["choice", "argument"],},
        ],)
    return completion
