import os
import re
import torch
import time
from tqdm import tqdm
import pandas as pd
from evaluate import load
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Print the device being used
print(f"Using device: {device}")

# Load your model and send it to the correct device
model_path = "ibm-granite/granite-3.0-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()

ds = load_dataset(
    "UWV/Leesplank_NL_wikipedia_simplifications_preprocessed", split='test')
df = ds.to_pandas()

df_size = len(df)
sari = load("sari")

# Split dataset into three parts
df_0 = df[:int(df_size*(1/3))]
# df_1 = df[int(df_size*(1/3)):int(df_size*(2/3))]
# df_2 = df[int(df_size*(2/3)):]

datasets = [df_0, df_1, df_2]
system_prompt = ("Vereenvoudig een Nederlandse alinea tot één duidelijke en aantrekkelijke tekst, "
                 "geschikt voor volwassenen die Nederlands als tweede taal spreken, met woorden uit de "
                 "'basiswoordenlijst Amsterdamse kleuters.' Behoud directe citaten, maak dialogen eenvoudiger "
                 "en leg culturele verwijzingen, uitdrukkingen en technische termen natuurlijk uit in de tekst. "
                 "Pas de volgorde van informatie aan voor betere eenvoud, aantrekkelijkheid en leesbaarheid. "
                 "Probeer geen komma’s of verkleinwoorden te gebruiken.")

sari_scores, sari_scores_avg = [], []
sari_errors = 0

# Start the overall timer
start_time_total = time.time()

for idx, data in enumerate(datasets):
    start_time_data = time.time()  # Timer for each dataset

    print(f"Starting benchmarking on dataset {idx + 1} (size: {len(data)})")

    sari_scores_data = []

    # Use tqdm for a progress bar over rows in the dataset
    for row_idx, row in tqdm(data.iterrows(), desc=f"Processing dataset {idx + 1}"):
        if row_idx > len(data)/2:
            print('Halfway there!')

        prompt = row['prompt']
        template = [
            # Adding system instruction
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},  # Your existing user prompt
        ]

        # Tokenize the input and move it to the selected device
        chat = tokenizer.apply_chat_template(
            template, tokenize=False, add_generation_prompt=True)
        input_tokens = tokenizer(chat, return_tensors="pt").to(device)

        # Generate output tokens
        output = model.generate(**input_tokens, max_new_tokens=500)

        # Decode output tokens into text
        output_text = tokenizer.batch_decode(output)

        # Extract the output using regex
        match = re.search(
            r'<\|start_of_role\|>assistant<\|end_of_role\|>(.*?)<\|end_of_text\|>', output_text[0])
        if match:
            output = match.group(1).replace('\"', '').replace('\\', '')

        # Compute SARI score
        try:
            sari_score = sari.compute(sources=[prompt], predictions=[
                                      output], references=[[row['result']]])['sari']
            sari_scores_data.append(sari_score)
        except ValueError as e:
            sari_errors += 1
            print(f'Error: {e}')
            print(f'This originates from the following row: {prompt}')

    # Store the individual dataset scores and average
    sari_scores.append(sari_scores_data)
    sari_scores_avg.append(sum(sari_scores_data) / len(sari_scores_data))

    # Print time taken for each dataset
    elapsed_time_data = time.time() - start_time_data
    print(f"Finished dataset {idx + 1} in {elapsed_time_data:.2f} seconds")

# Final time and results
elapsed_time_total = time.time() - start_time_total
print(f"Processing completed in {elapsed_time_total:.2f} seconds")
print(f"SARI scores averages: {sari_scores_avg}")
print(f"There were {sari_errors} rows that gave an error.")
