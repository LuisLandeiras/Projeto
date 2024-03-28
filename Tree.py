import spacy

def get_max_depth(token, depth=0):
    # Base case: If token has no children, return the current depth
    if not list(token.children):
        return depth
    # Recursive case: Calculate depth for each child
    else:
        return max(get_max_depth(child, depth + 1) for child in token.children)

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "The quick brown fox jumps over the lazy dog."

# Process the text with SpaCy
doc = nlp(text)

# Get the maximum depth of the dependency tree
max_depth = max(get_max_depth(sent.root) for sent in doc.sents)

print("Maximum depth of the dependency tree:", max_depth)
