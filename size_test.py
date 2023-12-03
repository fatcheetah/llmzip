import brotli

text = """
HN's content guidelines place no restriction on comments to be on-topic, let alone decide what on-topic is.

Do you really think there needs to be a guideline saying "comments should be on-topic"? Wouldn't you assume that based on both the guidelines and how we generally conduct ourselves in the comments? The guidelines don't say anything specifically about racism but surely you would agree that racist comments have no place here, right?
"""

compressed = brotli.compress(text.encode('utf-8'))

# size of original ?
print(len(text.encode('utf-8')))

# size of compressed ?
print(len(compressed))