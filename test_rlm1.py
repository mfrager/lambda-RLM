#!/usr/bin/env python3

import os
from rlm import LambdaRLM

with open("plato.txt") as f:
    document = f.read()

print("Document length: " + str(len(document)))

prompt = f"""Context:
{document}

Question: Summarize the main 10 ideas discussed in this document.

Answer:"""

rlm = LambdaRLM(
    backend_kwargs={
        "model_name": "",
        "api_key": "",
        "base_url": "",
    },
    context_window_tokens=30_000,
    verbose=True,
)

result = rlm.completion(prompt)
print(result.response)
