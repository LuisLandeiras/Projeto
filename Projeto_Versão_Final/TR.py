import spacy, AuxFun, time
import gensim
from gensim import corpora

t = time.process_time()
# Load the spaCy English model
nlp = spacy.load("en_core_web_lg")

Amostra = AuxFun.Amostras(AuxFun.File("TNasa.txt"))

# Preprocessing function using spaCy
def preprocess(text):
    # Process the text using spaCy
    doc = nlp(text)
    # Keep only the tokens that are not stop words and are alphabetic
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

# Apply preprocessing to the documents
texts = [preprocess(doc) for doc in Amostra]

# Create a dictionary and corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Train LDA model
lda_model = gensim.models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

# Display topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")

print(time.process_time() - t)