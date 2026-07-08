# Model Inference Boilerplate — Gemma 4 (E2B / E2B-it)
#
# Loads Google's `gemma-4-E2B` (base) and `gemma-4-E2B-it` (instruction-tuned)
# models and runs text generation on both, so you can compare their behavior.

# 1. Import Libraries
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import pandas as pd
import time
from tqdm import tqdm

# 2. Load Models and Processors
# Loads both the base (pre-trained) and instruction-tuned variants.
BASE_MODEL_ID = "google/gemma-4-E2B"
IT_MODEL_ID = "google/gemma-4-E2B-it"

base_processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype="auto",
    device_map="auto",
)

it_processor = AutoProcessor.from_pretrained(IT_MODEL_ID)
it_model = AutoModelForCausalLM.from_pretrained(
    IT_MODEL_ID,
    dtype="auto",
    device_map="auto",
)

# 3. Define Gemma Response Function
@torch.no_grad()
def generate_it_response(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    text = it_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = it_processor(text=text, return_tensors="pt").to(it_model.device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = it_model.generate(**inputs, max_new_tokens=256)
    response = it_processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    it_response = it_processor.parse_response(response)

    return it_response


# 4. Load Dataset

df = pd.read_excel("../data/eduBotSampleBulgarian30.xlsx")

# Your dataset columns are:
# ['id', 'subject', 'grade_level', 'topic', 'student_input',
#  'tutor_response', 'data_type', 'intent']

# 5. Generate Responses for Each Prompt

start_time = time.time()
gemma_responses = []

prompts = df["student_input"]
subjects = df["subject"]
grades = df["grade_level"]

gemma_responses_1 = []

for prompt in tqdm(prompts, desc="Generating Gemma responses 1"):
    response = generate_it_response(str(prompt))
    gemma_responses_1.append(response)

df["gemma_response_1"] = gemma_responses_1

gemma_responses_2 = []

for prompt, subject in tqdm(zip(prompts, subjects), total=len(df), desc="Generating Gemma responses 2"):
    full_prompt = f"{prompt}\n\nEmphasize topics in {subject}."
    response = generate_it_response(full_prompt)
    gemma_responses_2.append(response)

df["gemma_response_2"] = gemma_responses_2

gemma_responses_3 = []

for prompt, subject, grade in tqdm(zip(prompts, subjects, grades), total=len(df), desc="Generating Gemma responses 3"):
    full_prompt = f"{prompt}\n\nEmphasize topics in {subject} for a {grade} school level."
    response = generate_it_response(full_prompt)
    gemma_responses_3.append(response)

df["gemma_response_3"] = gemma_responses_3

end_time = time.time()
elapsed_time = end_time - start_time

minutes = int(elapsed_time // 60)
seconds = elapsed_time % 60


# 6. Save Results

df.to_excel("../results/eduBotBulgarian30_gemma_responses.xlsx", index=False)

print("Done. Saved results to ...")
print(f"Total generation time: {minutes} minutes and {seconds:.2f} seconds")
print(f"Average time per row: {elapsed_time / len(df):.2f} seconds")