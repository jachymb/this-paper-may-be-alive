# /// script
# requires-python = ">=3.12"
# dependencies = ["transformers", "torch"]
# ///
# usage: uv run llm.py 'Say \"I am alive\"' [tokens]
# Greedy decoding. Without a token count it runs until the model emits EOS.
# (Windows PowerShell 5.1 strips inner double quotes from arguments; escape them as \" as above.)
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, logging

MODEL = "C10X/checkpoint-27564"  # 6.4M params, 8 layers: smallest Hub model found to answer 'Say "I am alive"' (as a raw prompt, no chat template)

logging.set_verbosity_error()
logging.disable_progress_bar()
sys.stdout.reconfigure(encoding="utf-8")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
ids = tok(sys.argv[1], return_tensors="pt").input_ids  # the prompt as-is: this tokenizer adds no BOS and none is added here
n = int(sys.argv[2]) if len(sys.argv) > 2 else None
# ponytail: without a count the only stop besides EOS is the context window (8192 here); this model tends to loop and never emit EOS
limit = {"max_new_tokens": n, "min_new_tokens": n} if n else {"max_new_tokens": model.config.max_position_embeddings - ids.shape[1]}
model.generate(ids, do_sample=False, streamer=TextStreamer(tok, skip_prompt=True, skip_special_tokens=True), **limit)
