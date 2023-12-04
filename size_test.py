import brotli

text = """
It is a lossless text compression algorithm that uses large language models. Large lagnguage models are good at compression due to their natural ability to predict the next token in a sequence. LLMZip uses this ability to compress text. 
"""

compressed = brotli.compress(text.encode('utf-8'))

# size of original ?
print(len(text.encode('utf-8')))

# size of compressed ?
print(len(compressed))