# LLMZip

Thrown together implementation of LLMZip from the paper [https://arxiv.org/pdf/2306.04050.pdf](https://arxiv.org/pdf/2306.04050.pdf)

Different models will produce different results.
 Some information below regarding different models I've tested on the prompt below.


``` md
# prompt:
"""
BSD Zero Clause License

Copyright (c) [2023] [thefatcheetah]

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""

# model & outputs         : bytes
original text             : 672
text brotli compression   : 351

# llm tokens > brotli compression       : bytes
> openhermes-2.5-mistral-7b.Q4_K_M.gguf : 196
> openhermes-2.5-mistral-7b.Q2_K.gguf   : 187
> juanako-7b-una.Q4_K_M.gguf            : 170 *
> juanako-7b-una.Q2_K.gguf              : 179
```

<br>

+ advised possible to encode with arithmetic coding to reduce file size again.

<br>

```
LLMZip: Lossless Text Compression using Large Language Models - https://arxiv.org/abs/2306.04050 - Chandra Shekhara Kaushik Valmeekam, Krishna Narayanan, Dileep Kalathil, Jean-Francois Chamberland, Srinivas Shakkottai
```