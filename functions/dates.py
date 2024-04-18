# import spacy
# import re
# import numpy as np

# def generate_false_year(actual_year):
#     years = [year for year in range(1850, 2024) if year != actual_year]
#     return np.random.choice(years)

# def generate_false_month(actual_month):
#     months = ["January", "February", "March", "April", "May", "June",
#               "July", "August", "September", "October", "November", "December"]
#     filtered_months = [month for month in months if month != actual_month]
#     return np.random.choice(filtered_months)

# def safe_convert_to_int(s):
#     try:
#         return int(s)
#     except ValueError:
#         return None

# def extract_and_falsify_dates(text):
#     # Load the spaCy model
#     nlp = spacy.load("en_core_web_sm")
#     doc = nlp(text)
#     original_dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
#     falsified_dates = []

#     year_pattern = r'\b\d{4}\b'
#     months = ["January", "February", "March", "April", "May", "June",
#               "July", "August", "September", "October", "November", "December"]
    
#     for date in original_dates:
#         new_date = []
#         for part in date.split():
#             if part in months:
#                 new_date.append(generate_false_month(part))
#             elif re.match(year_pattern, part):
#                 year = safe_convert_to_int(part)
#                 if year:
#                     false_year = str(generate_false_year(year))
#                     new_date.append(false_year)
#                 else:
#                     new_date.append(part)  # Leave the part unchanged if it's not a valid year
#             else:
#                 new_date.append(part)
#         falsified_dates.append(" ".join(new_date))
    
#     return original_dates, falsified_dates

# def process_context(context):
#     original_dates, falsified_dates = extract_and_falsify_dates(context)
#     modified_context = context
#     for orig, fake in zip(original_dates, falsified_dates):
#         modified_context = re.sub(re.escape(orig), fake, modified_context)
#     return modified_context

import spacy
import re
import numpy as np
from tqdm import tqdm

