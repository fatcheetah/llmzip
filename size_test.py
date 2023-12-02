import brotli

text = """
The banana plant is the largest herbaceous flowering plant.[10] All the above-ground parts of a banana plant grow from a structure usually called a "corm".[11] Plants are normally tall and fairly sturdy with a treelike appearance, but what appears to be a trunk is actually a "false stem" or pseudostem. Bananas grow in a wide variety of soils, as long as the soil is at least 60 centimetres (2.0 ft) deep, has good drainage and is not compacted.[12] Banana plants are among the fastest growing of all plants, with daily surface growth rates recorded from 1.4 square metres (15 sq ft) to 1.6 square metres (17 sq ft).[13][14]
"""

compressed = brotli.compress(text.encode('utf-8'))

# size of compressed ?
print(len(compressed))

# size of original ?
print(len(text.encode('utf-8')))