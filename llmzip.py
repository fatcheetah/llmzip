# implementation of LLMZip by thefatcheetah
# LLMZip: Lossless Text Compression using LargeLanguage Models
# [https://arxiv.org/pdf/2306.04050.pdf](https://arxiv.org/pdf/2306.04050.pdf)

import numpy as np
from brotli import compress, decompress
from ctransformers import AutoModelForCausalLM

repo = "TheBloke/open-llama-3b-v2-wizard-evol-instuct-v2-196k-GGUF"
model = "open-llama-3b-v2-wizard-evol-instuct-v2-196k.Q3_K_S.gguf"

llm = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id=repo, 
                                           model_file=model,
                                           model_type="llama",
                                           context_length=1024)

prompt = """
This is a test of LLMZip. It is a lossless text compression algorithm that uses large language models. Large lagnguage models are good at compression due to their natural ability to predict the next token in a sequence. LLMZip uses this ability to compress text. It works by taking a sequence of 4 tokens and evaluating the sequence. The logits of the sequence are then used to determine the rank of the next token. The rank is then stored in a sequence of ranks. The sequence of ranks is then compressed using brotli compression. The sequence of ranks can then be decompressed and used to reconstruct the original text. This is a test of LLMZip. 
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
    llm.eval(tokens=sequence, batch_size=16)
    logits = llm.logits
    logits = np.array(logits, dtype=np.float32)

    # get the logit of the next token
    next_token_logit = logits[next_token]

    # create a mask where the value is True if the logit is greater than the next token's logit
    mask = logits > next_token_logit
    rank = np.count_nonzero(mask)
    sequence_of_ranks.append(rank)

llm.reset()

# save the sequence of ranks to a file (compressed) with brotli compression
array_sequence = np.array(sequence_of_ranks, dtype=np.uint32).tobytes()
compressed = compress(array_sequence)

print(f"\nbytes of original text: {len(prompt.encode('utf-8'))}")
print(f"bytes of tokens before compression: {len(array_sequence)}")
print(f"bytes of tokens after  compression: {len(compressed)}")

with open("compressed.bin", "wb") as f:
    f.write(compressed)

# we always start with the same 4 initial tokens as the loop starts on (epoch 5)
# TODO: we should store these in the file as well

# decompress the sequence of ranks from the file
with open("compressed.bin", "rb") as f:
    array_to_load = f.read()

decompressed = decompress(array_to_load)
loaded_sequence = np.frombuffer(decompressed, dtype=np.uint32)

new_tokens = tokens[:4]
for i in range(len(loaded_sequence)):
    sequence = new_tokens[-4:]

    # evaluate the sequence and store the logits
    llm.eval(tokens=sequence, batch_size=16)
    logits = llm.logits
    logits = np.array(logits)

    # get the indices of the logits sorted in descending order
    sorted_logits = np.argsort(logits)[::-1]
    value_of_rank = sorted_logits[loaded_sequence[i]]
    new_tokens.append(value_of_rank)
output_tokens = llm.detokenize(new_tokens)

print(output_tokens)
