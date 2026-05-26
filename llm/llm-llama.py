# =========================================================
# REAL LLAMA CHATBOT
# =========================================================
# Uses:
# - HuggingFace Transformers
# - Real Meta Llama model
# - GPU support
# - Streaming generation
# =========================================================

# =========================================================
# 1. INSTALL
# =========================================================
#
# pip install transformers accelerate torch sentencepiece
#
# =========================================================

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer
)

# =========================================================
# 2. MODEL CONFIG
# =========================================================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Other good options:
#
# "meta-llama/Llama-2-7b-chat-hf"
# "microsoft/phi-2"
# "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# "HuggingFaceH4/zephyr-7b-beta"

# =========================================================
# 3. DEVICE
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nUsing device: {device}")

# =========================================================
# 4. LOAD TOKENIZER
# =========================================================

print("\nLoading tokenizer...\n")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# =========================================================
# 5. LOAD MODEL
# =========================================================

print("\nLoading model...\n")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,

    torch_dtype=torch.float16
    if torch.cuda.is_available()
    else torch.float32,

    device_map="auto"
)

model.eval()

print("\nModel loaded successfully!")

# =========================================================
# 6. STREAMER
# =========================================================

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

# =========================================================
# 7. CHAT TEMPLATE
# =========================================================

def build_prompt(user_message):

    prompt = f"""
<|system|>
You are a helpful AI assistant.
</s>

<|user|>
{user_message}
</s>

<|assistant|>
"""

    return prompt

# =========================================================
# 8. GENERATE RESPONSE
# =========================================================

def chat(user_message,
         max_new_tokens=200,
         temperature=0.7,
         top_p=0.9):

    prompt = build_prompt(user_message)

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        output = model.generate(

            **inputs,

            max_new_tokens=max_new_tokens,

            temperature=temperature,

            top_p=top_p,

            do_sample=True,

            repetition_penalty=1.2,

            streamer=streamer,

            pad_token_id=tokenizer.eos_token_id
        )

# =========================================================
# 9. CHAT LOOP
# =========================================================

print("\n==============================")
print(" REAL LLAMA CHATBOT ")
print("==============================\n")

print("Type 'exit' to quit.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    print("\nAssistant:\n")

    chat(
        user_input,
        max_new_tokens=200,
        temperature=0.7
    )

    print("\n")
