# LLMZip

Thrown together implementation of LLMZip from the paper [https://arxiv.org/pdf/2306.04050.pdf](https://arxiv.org/pdf/2306.04050.pdf)

<br>

It is a lossless text compression algorithm that uses large language models. Large language models are good at compression due to their natural ability to predict the next token in a sequence. LLMZip uses this ability to compress text. 

It works by tokenizing the input taking a sequence of the first 4 tokens and evaluating the generated logits for the next token. 
The logits are then sorted and used to determine the rank of the next token. Storring each rank in a sequence of ranks. Taking the original 4 tokens and the generated sequence of ranks this can then be compressed with traditional methods.

 Decompression must be performed under the same model and conditions. Re-evaluating the original 4 tokens and selecting the logit by index from the sequence of ranks for that epoch. Storing this token into a token sequence which to be detokenized back into the original text lossless.

 <br>

**Different models will produce different results.**


``` md
# prompt:
"""
BSD Zero Clause License

Copyright (c) [2023] [thefatcheetah]

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""

# model & outputs       : bytes
original text           : 672
text brotli compression : 351

# sequence of ranks > brotli compression                   : bytes
> openhermes-2.5-mistral-7b.Q4_K_M.gguf                    : 196
> openhermes-2.5-mistral-7b.Q2_K.gguf                      : 187
> juanako-7b-una.Q2_K.gguf                                 : 179
> juanako-7b-una.Q4_K_M.gguf                               : 170
> open-llama-3b-v2-wizard-evol-instuct-v2-196k.Q3_K_S.gguf : 161 

# ranks variable length encdoded bytes > brotli compression : bytes
> open-llama-3b-v2-wizard-evol-instuct-v2-196k.Q3_K_S.gguf  : 123 * 
> RWKV-5-World-1B5-v2-20231025-ctx4096                      : 125  

## Super fast compression using RWKV! : seconds
* elapsed time                     : 3.10
* time per token                   : 0.0208

```

<br>

## cool stuff
 
### LLMZip Paper
 Lossless Text Compression using Large Language Models - [https://arxiv.org/abs/2306.04050](https://arxiv.org/abs/2306.04050) - Chandra Shekhara Kaushik Valmeekam, Krishna Narayanan, Dileep Kalathil, Jean-Francois Chamberland, Srinivas Shakkottai

<br>

### RWKV
[https://github.com/BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM)

RWKV is an RNN with Transformer-level LLM performance, which can also be directly trained like a GPT transformer (parallelizable). And it's 100% attention-free. You only need the hidden state at position t to compute the state at position t+1. You can use the "GPT" mode to quickly compute the hidden state for the "RNN" mode.


 <br>

 ### Jake VanderPlas on boolean arrays and masks
 [https://jakevdp.github.io/PythonDataScienceHandbook/02.06-boolean-arrays-and-masks.html](https://jakevdp.github.io/PythonDataScienceHandbook/02.06-boolean-arrays-and-masks.html)

 <br>


### What is variable-length integer encoding?

Variable-length integer encoding is a method of encoding integers in a way that uses less space for smaller numbers. It's often used in file formats and data storage for more efficient use of space.

One common form of variable-length integer encoding is "Base 128 Varint" used in Google's Protocol Buffers. In this encoding, each byte of the integer is stored in 7 bits of an output byte, and the 8th bit is used as a continuation bit. The continuation bit is set to 1 for all bytes except the last, which signals the end of the number.