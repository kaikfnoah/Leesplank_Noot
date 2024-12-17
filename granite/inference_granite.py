import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f'cuda:{local_rank}')
torch.cuda.set_device(local_rank)

print(device)

# Load your model and send it to the correct device
model_path = "ibm-granite/granite-3.0-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()

# change input text as desired
chat = [
    {"role": "user", "content": "Simplify: Jan Waaijer is getrouwd en heeft twee kinderen."},
]
chat = tokenizer.apply_chat_template(
    chat, tokenize=False, add_generation_prompt=True)

# tokenize the text
input_tokens = tokenizer(chat, return_tensors="pt")

# generate output tokens
output = model.generate(**input_tokens,
                        max_new_tokens=100)

# decode output tokens into text
output = tokenizer.batch_decode(output)

# print output
print(output)
