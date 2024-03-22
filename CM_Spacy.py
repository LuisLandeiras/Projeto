import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

#Complexity Measures
def File(File):
    with open(File, "r", encoding='utf-8', errors='ignore') as file:
        sentence = file.read().replace("\n", " ")
        
    return sentence

def SentenceLength(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    total_words = sum(len(sent.split()) for sent in sentences)
    avg_sentence_length = total_words / len(sentences)
    return avg_sentence_length

def WordLength(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha]  # Filter out non-alpha tokens
    total_chars = sum(len(token) for token in tokens)  # Calculate total characters
    avg_word_length = total_chars / len(tokens)
    return avg_word_length

def LexicalDensity(text): #  numero de palavras lexicais / Numero total de palavras
    doc = nlp(text.lower())
    content_words = []
    for token in doc:
        if token.pos_ == "ADP" or token.pos_ == "AUX" or token.pos_ == "PART" or token.pos_ == "CCONJ" or token.pos_ == "NOUN" or token.pos_ == "INTJ" or token.pos_ == "PROPN" or token.pos_ == "ADJ":
            content_words.append(token)
    lexical_density = len(content_words) / len(doc)
    return lexical_density

def LexicalDiversity(text): # numero de palavras diferentes / Numero total de palavras
    doc = nlp(text.lower())
    content_words = [token.text for token in doc if token.is_alpha]
    diff_words = list(dict.fromkeys(content_words))
    lexical_diversity = len(diff_words) / len(content_words) 
    return lexical_diversity

text = File("Textos/obama.txt")

print("Average Sentence Length:", SentenceLength(text))
print("Average Word Length:", WordLength(text))
print("Lexical Density:", LexicalDensity(text))
print("Lexical Diversity:", LexicalDiversity(text))
