# implementation of LLMZip by thefatcheetah
# LLMZip: Lossless Text Compression using LargeLanguage Models
# [https://arxiv.org/pdf/2306.04050.pdf](https://arxiv.org/pdf/2306.04050.pdf)

import torch
from ctransformers import AutoModelForCausalLM


llm = AutoModelForCausalLM.from_pretrained("TheBloke/OpenHermes-2.5-Mistral-7B-GGUF",
                                            model_file="openhermes-2.5-mistral-7b.Q4_K_M.gguf",
                                            model_type="mistral",
                                            context_length=2048,
                                            reset = True,
                                            gpu_layers=50)

prompt = """
The banana plant is the largest herbaceous flowering plant.[10] All the above-ground parts of a banana plant grow from a structure usually called a "corm".[11] Plants are normally tall and fairly sturdy with a treelike appearance, but what appears to be a trunk is actually a "false stem" or pseudostem. Bananas grow in a wide variety of soils, as long as the soil is at least 60 centimetres (2.0 ft) deep, has good drainage and is not compacted.[12] Banana plants are among the fastest growing of all plants, with daily surface growth rates recorded from 1.4 square metres (15 sq ft) to 1.6 square metres (17 sq ft).[13][14]
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
    sorted = torch.argsort(logits, descending=True).tolist()

    # using the logits, get the rank of the next token
    try:
        next_token = tokens[i+4]
    except IndexError:
        continue

    rank = sorted.index(next_token) 
    sequenceOfRanks.append(rank)

llm.reset()

# save the sequence of ranks to a file (compressed) with brotli compression

import array
import brotli

# some of the ranks are > 255, so we need to use a different type

arrayToSave = array.array('L', sequenceOfRanks).tobytes()

# could also not save the file but it's fun to store it

compressedArray = brotli.compress(arrayToSave)
print(f"\n bytes of original text : {len(prompt.encode('utf-8'))}")
print(f"bytes of compressed text : {len(compressedArray)}")

with open("compressed.bin", "wb") as f:
    f.write(compressedArray)

# decompress the sequence of ranks from the file

with open("compressed.bin", "rb") as f:
    arrayToLoad = f.read()

arrayToLoad = brotli.decompress(arrayToLoad)
loadedRanks = array.array('L')
loadedRanks.frombytes(arrayToLoad)

# we always start with the same 4 initial tokens as the loop starts on (epoch 5)
# TODO: we should store these in the file as well

newTokens = tokens[:4]

for i in range(len(loadedRanks)):
    sequence = newTokens[-4:]

    # evaluate the sequence
    llm.eval(tokens=sequence)
    logits = llm.logits

    # sort the probabilities
    logits = torch.tensor(logits, device="cuda:0")
    sorted = torch.argsort(logits, descending=True)

    valueOfRank = sorted[loadedRanks[i]]
    newTokens.append(valueOfRank)

output = llm.detokenize(newTokens)
print(output)
