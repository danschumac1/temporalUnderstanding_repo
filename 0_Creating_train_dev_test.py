#region # DATE INFO
"""
Created on 04/15/2024

@author: Dan
"""
#endregion
#region # IMPORTS
import os
set_path = '/home/dan/TemporalUnderstandingInLLMs'
os.chdir(set_path)

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import spacy 
import re
from concurrent.futures import ProcessPoolExecutor

# homebrewed functions
from functions.homebrew import extract_output_values_from_json_file


# REL
rel_output = extract_output_values_from_json_file('./data/output/rel_context.out')

#endregion
#region # TQ
# =============================================================================
# TQ
# =============================================================================
TQ_explicet = pd.read_csv('./data/TQ/TQ_explicet.csv', encoding='ISO-8859-1')
# TQ_explicet.loc(:'rel_C') = rel_output
TQ_explicet['relevant_context'] = rel_output
TQ_explicet['source'] = 'TQ_explicet_explicet'
TQ_explicet.reset_index(inplace=True)
TQ_explicet['idx'] = TQ_explicet['index'].apply(lambda x: 'TQ_exp_' + str(x))
TQ_explicet.drop('index', axis=1, inplace=True)  # Drop the 'index' column now that 'idx' is created

TQ_explicet = TQ_explicet.rename(columns={'index':'idx', 'Question':'question','Answer':'answer'})
# This dataset doesn't have  predefined train / dev / test splits

# we will create a 70 15 15 split. (350, 75,75 obs)
TQ_explicet_train, temp_df = train_test_split(TQ_explicet, test_size=0.3, random_state=27)
TQ_explicet_dev, TQ_explicet_test = train_test_split(temp_df, test_size=0.5, random_state=27)

#endregion
#region # TSQA
# =============================================================================
# TSQA
# =============================================================================
# EASY
TQ_explicet_train.reset_index(drop=True, inplace=True)

# train has 14308 obs
TSQA_train_easy = pd.read_json('./data/TSQA/easy/train_easy.json', lines=True)
# dev has 3021 obs
TSQA_dev_easy = pd.read_json('./data/TSQA/easy/dev_easy.json', lines=True)
# test has 2997 obs
TSQA_test_easy = pd.read_json('./data/TSQA/easy/test_easy.json', lines=True)

# HARD
# train has 14681 obs
TSQA_train_hard = pd.read_json('./data/TSQA/hard/train_hard.json', lines=True)
# dev has 3087 obs
TSQA_dev_hard = pd.read_json('./data/TSQA/hard/dev_hard.json', lines=True)
# test has 3078 obs
TSQA_test_hard = pd.read_json('./data/TSQA/hard/test_hard.json', lines=True)

# label the source of the each observation. (may help detangle later)
TSQA_easy = [TSQA_train_easy, TSQA_dev_easy, TSQA_test_easy]
for df in TSQA_easy:
    df['source'] = 'TSQA_easy'

# label the source of the each observation. (may help detangle later)
TSQA_hard = [TSQA_train_hard, TSQA_dev_hard, TSQA_test_hard]
for df in TSQA_hard:
    df['source'] = 'TSQA_hard'

TSQA = TSQA_easy + TSQA_hard
for df in TSQA:
    df.rename(columns={'context': 'relevant_context', 'targets': 'answer'}, inplace=True)
    cols_to_drop = ['from','end','paragraphs']
    for col in cols_to_drop:
        try:
            df.drop(col,inplace=True, axis=1)
        except:
            pass

#endregion
#region # Stacking Train, dev, test datasets.
# =============================================================================
# Stacking Train, dev, test datasets.
# =============================================================================
# TRAIN
train = pd.concat([TQ_explicet_train, TSQA_train_easy, TSQA_train_hard], ignore_index=True)
train['type'] = 'train'

# DEV
dev = pd.concat([TQ_explicet_dev, TSQA_dev_easy, TSQA_dev_hard], ignore_index=True)
dev['type'] = 'dev'

# TEST
test = pd.concat([TQ_explicet_test, TSQA_test_easy, TSQA_test_hard], ignore_index=True)
test['type'] = 'test'




#endregion
#region # RANDOM CONTEXT
# =============================================================================
# RANDOM CONTEXT
# =============================================================================
train['random_context'] = train['relevant_context'].shift(-1)
train.loc[0, 'random_context'] = train['relevant_context'].iloc[-1]

dev['random_context'] = dev['relevant_context'].shift(-1)
dev.loc[0, 'random_context'] = dev['relevant_context'].iloc[-1]

test['random_context'] = test['relevant_context'].shift(-1)
test.loc[0, 'random_context'] = test['relevant_context'].iloc[-1]

#endregion
#region # MIS-DATE-IFY
# =============================================================================
# MIS-DATE-IFY
# =============================================================================
nlp = None

def init_nlp():
    global nlp
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'attribute_ruler'])

def process_context(context):
    global nlp
    original_dates, falsified_dates = extract_and_falsify_dates(context)
    modified_context = context
    for orig, fake in zip(original_dates, falsified_dates):
        modified_context = re.sub(re.escape(orig), fake, modified_context)
    return modified_context

def generate_false_year(actual_year):
    years = np.setdiff1d(np.arange(1850, 2024), np.array([actual_year]))
    return np.random.choice(years)

def generate_false_month(actual_month):
    months = np.array(["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"])
    filtered_months = months[months != actual_month]
    return np.random.choice(filtered_months)

def safe_convert_to_int(s):
    try:
        return int(s)
    except ValueError:
        return None

def extract_and_falsify_dates(text):
    global nlp
    doc = nlp(text)
    original_dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    falsified_dates = []

    year_pattern = re.compile(r'\b\d{4}\b')
    for date in original_dates:
        new_date = []
        months = np.array(["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"])
        for part in date.split():
            if part in months:
                new_date.append(generate_false_month(part))
            elif year_pattern.match(part):
                year = safe_convert_to_int(part)
                if year:
                    false_year = str(generate_false_year(year))
                    new_date.append(false_year)
                else:
                    new_date.append(part)
            else:
                new_date.append(part)
        falsified_dates.append(" ".join(new_date))
    
    return original_dates, falsified_dates

# Process 'train' dataset
with ProcessPoolExecutor(max_workers=4, initializer=init_nlp) as executor:
    train_contexts = train['relevant_context']
    train_results = list(tqdm(executor.map(process_context, train_contexts), total=len(train_contexts)))
train['wrong_date_context'] = train_results

# Process 'dev' dataset
with ProcessPoolExecutor(max_workers=4, initializer=init_nlp) as executor:
    dev_contexts = dev['relevant_context']
    dev_results = list(tqdm(executor.map(process_context, dev_contexts), total=len(dev_contexts)))
dev['wrong_date_context'] = dev_results

# Process 'test' dataset
with ProcessPoolExecutor(max_workers=4, initializer=init_nlp) as executor:
    test_contexts = test['relevant_context']
    test_results = list(tqdm(executor.map(process_context, test_contexts), total=len(test_contexts)))
test['wrong_date_context'] = test_results

#endregion
#region # SAVE TO DISK
# =============================================================================
# SAVE TO DISK
# =============================================================================
train.to_csv('/home/dan/TemporalUnderstandingInLLMs/data/final/train.csv')
dev.to_csv('/home/dan/TemporalUnderstandingInLLMs/data/final/dev.csv')
test.to_csv('/home/dan/TemporalUnderstandingInLLMs/data/final/test.csv')
#endregion