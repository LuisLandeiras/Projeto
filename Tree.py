import nltk

# Sample text
text = "The quick brown fox jumps over the lazy dog."

# Tokenize the text into words
words = nltk.word_tokenize(text)

# Load a pre-trained PCFG model
grammar = nltk.data.load('grammars/large_grammars/atis.cfg')

# Create a parser with the PCFG grammar
parser = nltk.parse.pchart.InsideChartParser.fromstring(grammar)

# Generate parse trees
parse_trees = list(parser.parse(words))

# Display the parse trees
for tree in parse_trees:
    print(tree)

# Find the maximum depth of the parse trees
def max_depth(tree):
    if isinstance(tree, str):
        return 0
    else:
        depths = [max_depth(child) for child in tree]
        return max(depths) + 1

max_depths = [max_depth(tree) for tree in parse_trees]
print("Max Depth of the Trees:", max(max_depths))
