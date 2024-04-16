# =============================================================================
# Imports
# =============================================================================

# Homebrewed functions

# Standard imports
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
    TrainerCallback,
    DebertaV2Tokenizer

)
from copy import deepcopy
import csv


# =============================================================================
# LOAD DATA
# =============================================================================
TQ = pd.read_csv('./data/TQ_revised.csv')


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    print("This epoch's F1:", f1, "Acc:", acc)
    return {
        "f1_score": f1,
    }

def main(): 
    
    # =============================================================================
    # Load the data
    # =============================================================================
    
    
    train = sliding_window_divide_remainder('./data/train.csv',1)           
    dev = sliding_window_divide_remainder('./data/dev.csv',1) 
    
    # =============================================================================
    # Tokenize the Data
    # =============================================================================

    
    checkpoint = "nlpaueb/legal-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        truncation = True,
        padding = True,
        )
    
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    
    def cqa_tokenize_function(batch):
        return tokenizer(
            batch['cqa'],
            truncation=True,
            padding='max_length',
            max_length=512
        )
    
    tokenized_train = train.map(cqa_tokenize_function, batched=True)
    tokenized_train = tokenized_train.remove_columns(['cqa'])
    
    tokenized_dev = dev.map(cqa_tokenize_function, batched=True)
    tokenized_dev = tokenized_dev.remove_columns(['cqa'])
    
    
    # =============================================================================
    # Training
    # =============================================================================
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments( 
            'test-trainer_debertav3',
            learning_rate=3e-5,
            warmup_steps=10,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
          #  gradient_accumulation_steps = 2,
            num_train_epochs=50,
            fp16=True,
            save_strategy='epoch',
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            save_total_limit=1,
            greater_is_better=True, 
            metric_for_best_model="f1_score",
        )
    
    class CustomCallback(TrainerCallback):
        
        def __init__(self, trainer) -> None:
            super().__init__()
            self._trainer = trainer
        
        def on_epoch_end(self, args, state, control, **kwargs):
            if control.should_evaluate:
                control_copy = deepcopy(control)
                self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
                return control_copy
    
    
    #checkpoint = "DeBERTa-v3-base"
    
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    # trainer.add_callback(CustomCallback(trainer)) 
    
    trainer.train()
    
    
    # =============================================================================
    # Predictions
    # =============================================================================
    
    predictions = trainer.predict(tokenized_dev)
    preds_dict = {}
    labels_dict = {}
    
    for preds,row,labs in zip(predictions.predictions, tokenized_dev, predictions.label_ids):
        
        idx = int(row['idx'])
        labels_dict[idx] = int(labs)
        if idx in preds_dict:
            preds_dict[idx].append(preds)
        else:
            preds_dict[idx] = [preds]
    
    probs_list = []
    labs_list = []
    
    for k,v in preds_dict.items():
        labs_list.append(labels_dict[k])
        current_val = v[0]
        count = 1
        for x in v[1:]:
            current_val += x
            count += 1
        current_val /= count
        probs_list.append(current_val)
    
    probs_list = np.array(probs_list)
    
    preds = np.argmax(probs_list, axis=-1)
    f1_bin = f1_score(labs_list, preds)
    print("F1 Score binary:", f1_bin)
    f1_mac = f1_score(labs_list, preds, average="macro")
    print('F1 Score macro:', f1_mac)
    
if __name__ == '__main__':
    main()
