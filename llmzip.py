# implementation of LLMZip by thefatcheetah
# LLMZip: Lossless Text Compression using LargeLanguage Models
# [https://arxiv.org/pdf/2306.04050.pdf]

import brotli
import numpy as np
from ctransformers import AutoModelForCausalLM

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


repo = "TheBloke/open-llama-3b-v2-wizard-evol-instuct-v2-196k-GGUF"
model = "open-llama-3b-v2-wizard-evol-instuct-v2-196k.Q3_K_S.gguf"

llm = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id=repo, 
                                           model_file=model,
                                           model_type="llama",
                                           context_length=1024)

prompt = """
BSD Zero Clause License

Copyright (c) [2023] [thefatcheetah]

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""

tokens = llm.tokenize(text=prompt, add_bos_token=False)

sequence_of_ranks = [] 
for i in range(len(tokens) - 3):
    # take a sequence of 4 tokens from current i 
    sequence = tokens[i:i+4]

    # get the rank of the next token
    try:
        next_token = tokens[i+4]
    except IndexError:
        continue

    # evaluate the sequence and store the logits
    llm.eval(tokens=sequence)
    logits = llm.logits
    logits = np.array(logits, dtype=np.float32)

    # get the logit of the next token
    next_token_logit = logits[next_token]

    # create a mask where the value is True if the logit is greater than the next token's logit
    mask = logits > next_token_logit
    rank = np.count_nonzero(mask)
    sequence_of_ranks.append(encode_varint(rank))

llm.reset()

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


new_tokens = tokens[:4]
for i in range(len(loaded_sequence)):
    sequence = new_tokens[-4:]

    # evaluate the sequence and store the logits
    llm.eval(tokens=sequence)
    logits = llm.logits
    logits = np.array(logits, dtype=np.float32)

    # get the indices of the logits sorted in descending order
    sorted_logits = np.argsort(logits)[::-1]
    value_of_rank = sorted_logits[loaded_sequence[i]]
    new_tokens.append(value_of_rank)
output_tokens = llm.detokenize(new_tokens)

print(output_tokens)
