import spacy, AuxFun, time
from gensim.models import KeyedVectors
from collections import defaultdict

t = time.process_time()

# Load pre-trained word vectors (use the appropriate model path)
word_vectors = KeyedVectors.load_word2vec_format('C:/Users/luis_/OneDrive/Ambiente de Trabalho/GoogleNews-vectors-negative300.bin', binary=True)

# Load the spaCy English model
nlp = spacy.load("en_core_web_md")

def get_lexical_chains(text):
    # Process the text with spaCy
    doc = nlp(text)

    # Create a dictionary to store lexical chains
    lexical_chains = defaultdict(list)

    for token in doc:
        # Consider only nouns and verbs for lexical chains
        if token.pos_ in ['NOUN', 'VERB']:
            try:
                # Find similar words based on cosine similarity
                similar_words = word_vectors.most_similar(token.text, topn=5)
                for similar_word, _ in similar_words:
                    lexical_chains[token.text].append(similar_word)
            except KeyError:
                # If the token is not in the word vector model, skip it
                continue

    return lexical_chains


Amostra = AuxFun.File("TNasa.txt")

# Get lexical chains
chains = get_lexical_chains(Amostra)

# Print the lexical chains
for word, similar_words in chains.items():
    print(f"{word}: {', '.join(similar_words)}")
    
print(time.process_time() - t)
