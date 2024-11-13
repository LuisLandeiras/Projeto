#Text Complexity Measures
import spacy, time, syllapy

nlp = spacy.load('en_core_web_sm')

def Normalize(Valor, Max, Min):
    return (Valor - Min) / (Max - Min)

# Funções com o texto completo
def SentenceLength(text):
    t = time.process_time()
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    TotalWords = sum(len(sent.split()) for sent in sentences)
    SentenceL = TotalWords / len(sentences)
    return round(SentenceL,3), time.process_time() - t

def WordLength(text):
    t = time.process_time()
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha]
    TotalChars = sum(len(token) for token in tokens)
    WordL = TotalChars / len(tokens)
    return round(WordL,3), time.process_time() - t

def LexicalDensity(text): 
    t = time.process_time()
    doc = nlp(text.lower())
    LexicalWords = []
    for token in doc:
        if token.pos_ == "ADP" or token.pos_ == "AUX" or token.pos_ == "PART" or token.pos_ == "CCONJ" or token.pos_ == "NOUN" or token.pos_ == "INTJ" or token.pos_ == "PROPN" or token.pos_ == "ADJ":
            LexicalWords.append(token)
    LDensity = len(LexicalWords) / len(doc)
    return round(LDensity,3), time.process_time() - t

def LexicalDiversity(text): # Numero de palavras diferentes / Numero total de palavras
    t = time.process_time()
    doc = nlp(text.lower())
    AlphaWords = [token.text for token in doc if token.is_alpha]
    DiffWords = list(dict.fromkeys(AlphaWords))
    LDiversity = len(DiffWords) / len(AlphaWords)
    return round(LDiversity,3), time.process_time() - t
#--------------------------------------------------------

#Funções com amostras
def LexicalDensityA(Samples): # Numero de palavras lexicais / Numero total de palavras uyvu
    t = time.process_time()
    Soma = 0
    Count = 0
    for Sample in Samples:
        doc = nlp(str(Sample).lower())
        lexical_words = [token.text for token in doc if token.pos_ in {"ADP", "AUX", "PART", "CCONJ", "NOUN", "INTJ", "PROPN", "ADJ"}]
        Count += len(str(Sample).split())
        Soma += len(lexical_words)
    
    return round(Soma/Count,3), time.process_time() - t

def LexicalDiversityA(Samples): # Numero de palavras diferentes / Numero total de palavras(TTR)
    t = time.process_time()    
    Text = ' '.join(Samples)
    Words = Text.split()
    return round(len(set(Words))/len(Words),3), time.process_time() - t

def WordLengthA(Samples):
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        Sentence = str(Sample).split()
        WordLength = sum(len(Amostra) for Amostra in Sentence)
        Soma += WordLength/len(Sentence)
        
    Normalizacao = Normalize(Soma/len(Samples),7,3)
    
    return round(Soma/len(Samples),3), round(Normalizacao,3), time.process_time() - t

def SentenceLengthA(Samples):
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        Texto = str(Sample)
        TotalWords = len(Texto.split())
        Soma += TotalWords
        
    Normalizacao = Normalize(Soma/len(Samples),100,4)
    
    return round(Soma/len(Samples),3), round(Normalizacao,3), time.process_time() - t

def SyllableAve(Samples):
    t = time.process_time()
    Syllables = 0
    Words = 0
    for Sample in Samples:
        doc = nlp(str(Sample))
        for token in doc:
            if token.is_alpha:
                Syllables += syllapy.count(token.text)
                Words += 1
    
    Soma = Syllables/Words
    Normalizacao = Normalize(Soma,2,1)
    
    return round(Soma,3), round(Normalizacao,3), time.process_time() - t
     