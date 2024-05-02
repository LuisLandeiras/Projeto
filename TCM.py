#Text Complexity Measures
import spacy, time

nlp = spacy.load('en_core_web_sm')

# Funções com o texto completo
def SentenceLength(text):
    t = time.process_time()
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    TotalWords = sum(len(sent.split()) for sent in sentences)
    SentenceL = TotalWords / len(sentences)
    return SentenceL, time.process_time() - t

def WordLength(text):
    t = time.process_time()
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha]
    TotalChars = sum(len(token) for token in tokens)
    WordL = TotalChars / len(tokens)
    return WordL, time.process_time() - t

def LexicalDensity(text): 
    t = time.process_time()
    doc = nlp(text.lower())
    LexicalWords = []
    for token in doc:
        if token.pos_ == "ADP" or token.pos_ == "AUX" or token.pos_ == "PART" or token.pos_ == "CCONJ" or token.pos_ == "NOUN" or token.pos_ == "INTJ" or token.pos_ == "PROPN" or token.pos_ == "ADJ":
            LexicalWords.append(token)
    LDensity = len(LexicalWords) / len(doc)
    return LDensity, time.process_time() - t

def LexicalDiversity(text): # Numero de palavras diferentes / Numero total de palavras
    t = time.process_time()
    doc = nlp(text.lower())
    AlphaWords = [token.text for token in doc if token.is_alpha]
    DiffWords = list(dict.fromkeys(AlphaWords))
    LDiversity = len(DiffWords) / len(AlphaWords)
    return LDiversity, time.process_time() - t
#--------------------------------------------------------

#Funções com amostras
def LexicalDensityA(Samples): # Numero de palavras lexicais / Numero total de palavras
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        doc = nlp(str(Sample).lower())
        lexical_words = [token.text for token in doc if token.pos_ in {"ADP", "AUX", "PART", "CCONJ", "NOUN", "INTJ", "PROPN", "ADJ"}]
        Soma += len(lexical_words)
    return '{:.3}'.format(Soma/10000), time.process_time() - t

def LexicalDiversityA(Samples): # Numero de palavras diferentes / Numero total de palavras
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        DiffWords = list(dict.fromkeys(Sample))
        Soma += len(DiffWords)
    return '{:.3}'.format(Soma/10000), time.process_time() - t

def WordLengthA(Samples):
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        WordLength = sum(len(Amostra) for Amostra in Sample)
        Soma += WordLength
    return '{:.3}'.format(Soma/10000), time.process_time() - t

def SentenceLengthA(Samples):
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        Texto = str(Sample)
        TotalWords = len(Texto.split())
        Soma += TotalWords/1
    return '{:.3}'.format(Soma/len(Samples)), time.process_time() - t
