# implementation of LLMZip by thefatcheetah
# LLMZip: Lossless Text Compression using LargeLanguage Models
# [https://arxiv.org/pdf/2306.04050.pdf]

import brotli
import numpy as np
import os
import torch
from huggingface_hub import hf_hub_download
from rwkv.model import RWKV
from rwkv.utils import PIPELINE


# variable-length integer encoding
def encode_varint(number):
    bytes_list = []
    while number > 127:
        bytes_list.append((number & 127) | 128)
        number >>= 7
    bytes_list.append(number)
    return bytes(bytes_list)


def decode_varint(bytes_list):
    number = 0
    shift = 0
    for byte in bytes_list:
        number |= (byte & 127) << shift
        if byte & 128 == 0:
            break
        shift += 7
    return number


title = "RWKV-5-World-3B-v2-20231113-ctx4096"

os.environ["RWKV_JIT_ON"] = '1'
model_path = hf_hub_download(repo_id="BlinkDL/rwkv-5-world", filename=f"{title}.pth")
model = RWKV(model=model_path, strategy='cuda fp16')
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

prompt = """
BSD Zero Clause License

Copyright (c) [2023] [thefatcheetah]

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""

tokens = pipeline.encode(prompt)

sequence_of_ranks = []
# create an initial state for the model
for i in range(len(tokens) - 3):
    # take a sequence of 4 tokens from current i 
    sequence = tokens[i:i+4]

    # get the rank of the next token
    try:
        next_token = tokens[i+4]
    except IndexError:
        continue

    # evaluate the sequence and sort the logits
    out, state = model.forward(sequence, None)
    logits = out.detach().cpu().numpy()
    sorted_logits = np.argsort(logits)[::-1].tolist()

    # get the rank of the next token
    rank = sorted_logits.index(next_token)
    sequence_of_ranks.append(encode_varint(rank))

# save the sequence of ranks to a file (compressed) with brotli compression
bytes_of_ranks = b''.join(sequence_of_ranks)
compressed_bytes = brotli.compress(bytes_of_ranks)

print(f"\nbytes of original text: {len(prompt.encode('utf-8'))}")
print(f"bytes of tokens before compression: {len(bytes_of_ranks)}")
print(f"bytes of tokens after  compression: {len(compressed_bytes)}")

with open("compressed.bin", "wb") as f:
    f.write(compressed_bytes)

# # we always start with the same 4 initial tokens as the loop starts on (epoch 5)
# # TODO: we should store these in the file as well
new_tokens = tokens[:4]

# # decompress the sequence of ranks from the file
with open("compressed.bin", "rb") as f:
    array_to_load = f.read()

decompressed_bytes = brotli.decompress(array_to_load)

loaded_sequence = []
i = 0
while i < len(decompressed_bytes):
    # get the next varint from bytes_of_ranks
    varint_bytes = []
    while True:
        byte = decompressed_bytes[i]
        varint_bytes.append(byte)
        i += 1
        if byte & 128 == 0:
            break

    # decode the varint and add it to loaded_sequence
    rank = decode_varint(varint_bytes)
    loaded_sequence.append(rank)

for i in range(len(loaded_sequence)):
    sequence = new_tokens[-4:]

    # evaluate the sequence and sort the logits
    out, state = model.forward(sequence, None)
    logits = out.detach().cpu().numpy()

    # get the indices of the logits sorted in descending order
    sorted_logits = np.argsort(logits)[::-1]
    value_of_rank = sorted_logits[loaded_sequence[i]]
    new_tokens.append(value_of_rank)

output_tokens = pipeline.decode(new_tokens)
print(output_tokens)
