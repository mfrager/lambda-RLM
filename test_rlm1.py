#!/usr/bin/env python3

import os
from rlm import LambdaRLM
from dotenv import load_dotenv

load_dotenv()

with open("plato.txt") as f:
    document = f.read()

print("Document length: " + str(len(document)))

#MODEL = "lmstudio_z/gemma-4-e4b-it"
MODEL = "auto/local_z"

prompt = f"""Context:
{document}

Question: Summarize the main 10 ideas discussed in this document.

Answer:"""

rlm = LambdaRLM(
    backend_kwargs={
        "model_name": MODEL,
        "api_key": os.environ['OPENAI_API_KEY'],
        "base_url": os.environ['OPENAI_BASE_URL'],
    },
    context_window_tokens=10_000,
    verbose=True,
)

result = rlm.completion(prompt)
print(result.response)

