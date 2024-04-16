#region # IMPORTS
import pandas as pd
import json
import numpy as np
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

#endregion
#region # LOAD DATA
# =============================================================================
# LOAD DATA
# =============================================================================
# import os
# os.getcwd()
# TEMPORAL QUESTIONS
TQ = pd.read_csv('./data/TQ.csv', encoding='ISO-8859-1')
# not a very large dataset unfortunatly...
#endregion
#region # PROMPT HEAD AND PROMPT

# =============================================================================
# SET UP PROMPTHEAD AND PROMPT
# =============================================================================

TQ['PromptHead'] = [f'Question: {q}\n\nAnswer: {a}\n\Generated Context: ' for q, a in zip(TQ['Question'],TQ['Answer'])]

rel_prompt = '''
You will be provided with a question and an answer. Let's play a game where you pretend that all of the answers are correct. Your job is to output a *Generated Context* section. In this section, generate some relevant context that would help answer the question. The context should be in the same style and format as a wikipedia article.

Question: What Movie won seven Oscars in 1994?

Answer: Schindler's List

Generated Context: Schindler's List premiered on November 30, 1993, in Washington, D.C., and was released on December 15, 1993, in the United States. Often listed among the greatest films ever made, the film received widespread critical acclaim for its tone, acting (particularly Neeson, Fiennes, and Kingsley), atmosphere, score, cinematography, and Spielberg's direction; it was also a box office success, earning $322.2 million worldwide on a $22 million budget. It was nominated for twelve awards at the 66th Academy Awards, and won seven, including Best Picture, Best Director (for Spielberg), Best Adapted Screenplay, and Best Original Score. The film won numerous other awards, including seven BAFTAs and three Golden Globe Awards. In 2007, the American Film Institute ranked Schindler's List 8th on its list of the 100 best American films of all time. The film was designated as "culturally, historically or aesthetically significant" by the Library of Congress in 2004 and selected for preservation in the United States National Film Registry.


''' # Leave the spaces ^^ 

#endregion
#region # API CONFIG
# =============================================================================
# api config
# =============================================================================

# Load environment variables from the .env file
load_dotenv('./data/.env')

# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is available
if api_key is None:
    raise ValueError("API key is missing. Make sure to set OPENAI_API_KEY in your environment.")

# Set the API key for the OpenAI client
openai.api_key = api_key
client = OpenAI(api_key=api_key)
#endregion
#region # PROMPTING
# =============================================================================
# prompting
# =============================================================================
for i, promptHead in enumerate(TQ['PromptHead']):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
        {
            "role": "system",
            "content": '''
{rel_prompt}
'''
        },
        {
          "role": "user",
          "content": promptHead
        }
      ],
      temperature=0.7,
      max_tokens=1024,
      top_p=1
    )
        
    print(json.dumps({ 'i':i , 'output': response.choices[0].message.content}))  

# in linux command line run
# CUDA_VISIBLE_DEVICES=0 nohup python make_rel_context.py > ./data/output/rel_context.out &
#endregion
