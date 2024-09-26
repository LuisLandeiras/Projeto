import spacy

# Load a medium-sized language model that supports probabilities
nlp = spacy.load("en_core_web_lg")  # or "en_core_web_md"

# Sample text
text = "This is a sample text. This text is for NLP processing."

# Process the text
doc = nlp(text)

# Print token probabilities
for token in doc:
    if token.is_alpha:
        print(f"Word: {token.text}, Probability: {token.prob}")
