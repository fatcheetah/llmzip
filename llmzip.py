# implementation of LLMZip by thefatcheetah
# LLMZip: Lossless Text Compression using LargeLanguage Models
# [https://arxiv.org/pdf/2306.04050.pdf](https://arxiv.org/pdf/2306.04050.pdf)

import torch
from ctransformers import AutoModelForCausalLM

repo = "TheBloke/juanako-7B-UNA-GGUF"
model = "juanako-7b-una.Q4_K_M.gguf"

llm = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id=repo, 
                                           model_file=model,
                                           model_type="llama",
                                           context_length=2048,
                                           max_new_tokens=1,
                                           temperature=1,
                                           gpu_layers=40)

prompt = """
BSD Zero Clause License

Copyright (c) [2023] [thefatcheetah]

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""

tokens = llm.tokenize(text=prompt, add_bos_token=False)

sequenceOfRanks = [] 
for i in range(len(tokens) - 3):
    # take a sequence of 4 tokens from current i 
    sequence = tokens[i:i+4]

    # evaluate the sequence
    llm.eval(tokens=sequence)
    logits = llm.logits

    # sort the probabilities
    logits = torch.tensor(logits, device="cuda:0")
    sorted_logits = torch.argsort(logits, descending=True).tolist()

    # using the logits, get the rank of the next token
    try:
        next_token = tokens[i+4]
    except IndexError:
        continue

    rank = sorted_logits.index(next_token) 
    sequenceOfRanks.append(rank)

llm.reset()

# save the sequence of ranks to a file (compressed) with brotli compression

import array
import brotli

array_sequence = array.array('I', sequenceOfRanks).tobytes()

compressed = brotli.compress(array_sequence)

print(f"\nbytes of original text: {len(prompt.encode('utf-8'))}")
print(f"bytes of tokens before compression: {len(array_sequence)}")
print(f"bytes of tokens after  compression: {len(compressed)}")

with open("compressed.bin", "wb") as f:
    f.write(compressed)

# decompress the sequence of ranks from the file

with open("compressed.bin", "rb") as f:
    arrayToLoad = f.read()

decompressed = brotli.decompress(arrayToLoad)
loaded_sequence = array.array('I')
loaded_sequence.frombytes(decompressed)

# we always start with the same 4 initial tokens as the loop starts on (epoch 5)
# TODO: we should store these in the file as well

newTokens = tokens[:4]

for i in range(len(loaded_sequence)):
    sequence = newTokens[-4:]

    # evaluate the sequence
    llm.eval(tokens=sequence)
    logits = llm.logits

    # sort the probabilities
    logits = torch.tensor(logits, device="cuda:0")
    sorted_logits = torch.argsort(logits, descending=True)

    valueOfRank = sorted_logits[loaded_sequence[i]]
    newTokens.append(valueOfRank)

output = llm.detokenize(newTokens)
print(output)
