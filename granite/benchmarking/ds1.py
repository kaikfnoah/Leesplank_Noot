import re
import torch
import time
from tqdm import tqdm
from evaluate import load
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import InferenceEngine

# Step 1: Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Print the device being used
print(f"Using device: {device}")

# Step 2: Load the Hugging Face tokenizer and model
model_path = "ibm-granite/granite-3.0-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# We don't load the model directly using transformers here because vLLM will handle it
# Step 3: Initialize the vLLM Inference Engine with multi-GPU support

# Configure the device_map for 4 GPUs
# You can specify which GPUs you want to use (e.g., 0, 1, 2, 3)
device_map = {0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}

# Initialize the Inference Engine, specifying the model path and device_map for parallelism
inference_engine = InferenceEngine.from_pretrained(
    model_path,
    device_map=device_map,  # Specify the GPUs
    dtype=torch.float16,     # Use mixed-precision for better performance
    use_triton=True,         # Optionally enable Triton for optimized inference
    max_batch_size=16,       # Batch size per GPU (adjust based on GPU memory)
)

# Step 4: Prepare your inputs
text = "Write a detailed explanation of quantum mechanics."

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt").to(device)

# Step 5: Perform inference with vLLM's Inference Engine
# Use the inference engine to generate predictions (this will automatically use all GPUs)
output = inference_engine(inputs["input_ids"])

# Step 6: Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Step 7: Print the generated text
print(f"Generated Text: {generated_text}")

ds = load_dataset(
    "UWV/Leesplank_NL_wikipedia_simplifications_preprocessed", split='test')
df = ds.to_pandas()

df_size = len(df)
sari = load("sari")

# Split dataset into three parts
df_0 = df[:int(df_size*(1/3))]

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

print(f"Starting benchmarking on dataset 1 (size: {len(df_0)})")

sari_scores_data = []

# Use tqdm for a progress bar over rows in the dataset
# for row_idx, row in tqdm(df_0.iterrows()):
# if row_idx > len(df_0)/2:
#     print('Halfway there!')

# prompt = row['prompt']
prompt = "Hij is getrouwd met Erin Barrett Horowitz."
template = [
    {"role": "system", "content": system_prompt},  # Adding system instruction
    {"role": "user", "content": prompt},  # Your existing user prompt
]

# Tokenize the input and move it to the selected device
chat = tokenizer.apply_chat_template(
    template, tokenize=False, add_generation_prompt=True)
input_tokens = tokenizer(chat, return_tensors="pt").to(device)

# Generate output tokens
output = inference_engine(inputs["input_ids"])

# Step 6: Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

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
    print(f"Finished dataset 1 in {elapsed_time_data:.2f} seconds")

# Final time and results
elapsed_time_total = time.time() - start_time_total
print(f"Processing completed in {elapsed_time_total:.2f} seconds")
print(f"SARI scores averages: {sari_scores_avg}")
print(f"There were {sari_errors} rows that gave an error.")
