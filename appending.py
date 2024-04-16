#region # DATE INFO
"""
Created on 04/15/2024

@author: Dan
"""
#endregion
#region # IMPORTS
import pandas as pd
import json
import numpy as np
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
os.getcwd()
os.chdir('./TemporalUnderstandingInLLMs')
import json
from functions.homebrew import extract_output_values_from_json_file
from sklearn.model_selection import train_test_split

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
TQ_explicet.columns
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
train['type'] = 'dev'

# TEST
test = pd.concat([TQ_explicet_test, TSQA_test_easy, TSQA_test_hard], ignore_index=True)
train['type'] = 'test'

# ALL 
all_data = pd.concat([train, dev, test], ignore_index=True)

#endregion
#region # MIS-DATE-IFY
# =============================================================================
# MIS-DATE-IFY
# =============================================================================
import spacy
import numpy as np
import re

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def generate_false_year(actual_year):
    years = [year for year in range(1850, 2024) if year != actual_year]
    return np.random.choice(years)

def generate_false_month(actual_month):
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    filtered_months = [month for month in months if month != actual_month]
    return np.random.choice(filtered_months)

def safe_convert_to_int(s):
    try:
        return int(s)
    except ValueError:
        return None

def extract_and_falsify_dates(text):
    doc = nlp(text)
    original_dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    falsified_dates = []

    year_pattern = r'\b\d{4}\b'
    
    for date in original_dates:
        new_date = []
        for part in date.split():
            if part in months:
                new_date.append(generate_false_month(part))
            elif re.match(year_pattern, part):
                year = safe_convert_to_int(part)
                if year:
                    false_year = str(generate_false_year(year))
                    new_date.append(false_year)
                else:
                    new_date.append(part)  # Leave the part unchanged if it's not a valid year
            else:
                new_date.append(part)
        falsified_dates.append(" ".join(new_date))
    
    return original_dates, falsified_dates


from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def process_context(context):
    original_dates, falsified_dates = extract_and_falsify_dates(context)
    modified_context = context
    for orig, fake in zip(original_dates, falsified_dates):
        modified_context = re.sub(re.escape(orig), fake, modified_context)
    return modified_context

# Use ThreadPoolExecutor to parallelize processing
with ThreadPoolExecutor(max_workers=4) as executor:
    train_results = list(tqdm(executor.map(process_context, train['relevant_context']), total=len(train['relevant_context'])))
train['wrong_date_context'] = train_results


with ThreadPoolExecutor(max_workers=4) as executor:
    dev_results = list(tqdm(executor.map(process_context, dev['relevant_context']), total=len(dev['relevant_context'])))

dev['wrong_date_context'] = dev_results


with ThreadPoolExecutor(max_workers=4) as executor:
    test_results = list(tqdm(executor.map(process_context, test['relevant_context']), total=len(test['relevant_context'])))

test['wrong_date_context'] = test_results

#endregion
#endregion
#region # RANDOM CONTEXT
# =============================================================================
# RANDOM CONTEXT
# =============================================================================
import pandas as pd

# Assuming train, dev, and test are already defined DataFrames
dataframes = {
    'train': train,
    'dev': dev,
    'test': test
}

# Dictionary to store shuffled versions
shuffled_dataframes = {}

for name, df in dataframes.items():
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1, random_state=27).reset_index(drop=True)

    # Shift the 'relevant_context' to create 'random_context'
    df_shuffled['random_context'] = df_shuffled['relevant_context'].shift(-1)

    # Wrap around: setting the last element of 'random_context' to the first element of 'relevant_context'
    df_shuffled['random_context'].iloc[-1] = df_shuffled['relevant_context'].iloc[0]

    # Store the shuffled DataFrame
    shuffled_dataframes[name] = df_shuffled

    # Optionally print the modified DataFrame
    print(f"Modified {name}:")
    print(df_shuffled, "\n")


#region # SAVE TO DISK
# =============================================================================
# SAVE TO DISK
# =============================================================================
os.getcwd()
train.to_csv('./data/final/train.csv')
dev.to_csv('./data/final/dev.csv')
test.to_csv('./data/final/test.csv')