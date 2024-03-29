#Text Complexity Measures
import spacy, time, AuxFun

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

def LexicalDensity(text): # Numero de palavras lexicais / Numero total de palavras
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
        doc = nlp(str(Sample))
        for token in doc:
            if token.pos_ == "ADP" or token.pos_ == "AUX" or token.pos_ == "PART" or token.pos_ == "CCONJ" or token.pos_ == "NOUN" or token.pos_ == "INTJ" or token.pos_ == "PROPN" or token.pos_ == "ADJ":
                Soma += 1
    return Soma/10000, time.process_time() - t

def LexicalDiversityA(Samples): # Numero de palavras diferentes / Numero total de palavras
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        DiffWords = list(dict.fromkeys(Sample))
        Soma += len(DiffWords)
    return Soma/10000, time.process_time() - t

def WordLengthA(Samples):
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        WordLength = sum(len(Amostra) for Amostra in Sample)
        Soma += WordLength
    return Soma/10000, time.process_time() - t

def SentenceLengthA(Samples):
    t = time.process_time()
    SentenceLength = 0
    for i in range(len(Samples)):
        doc = nlp(Samples[i])
        sentences = [sent.text for sent in doc.sents]
        TotalWords = sum(len(sent.split()) for sent in sentences)
        SentenceLength += TotalWords / len(sentences)
    return SentenceLength/10, time.process_time() - t

#def TTRMedio(Samples):
#    Soma = 0
#    for Sample in Samples: #1000 samples são estudadas para o resultado final
#        Soma += len(set(Sample))/len(Sample) #Divisão entre a sample limpa(Palavras não repetidas) e o tamanho total da sample
#    return Soma/100
#--------------------------------------------------------

text = AuxFun.File("Textos/The_Mother.txt")
Samples = AuxFun.Amostras(text, 10)

SentenceL = SentenceLength(text) 
SentenceLA = SentenceLengthA(Samples[1])

print("Average Sentence Length:", SentenceL[0], "Time:", SentenceL[1])
print("Average Sentence Length Amostras:", SentenceLA[0], "Time:", SentenceLA[1])

print("--------------------------------------------")

WordL = WordLength(text)
WordLA = WordLengthA(Samples[0])

print("Average Word Length:", WordL[0], "Time:", WordL[1])
print("Average Word Length Amostras:", WordLA[0], "Time:", WordLA[1])

print("--------------------------------------------")

LDensity = LexicalDensity(text)
LDensityA = LexicalDensityA(Samples[0])

print("Lexical Density:", LDensity[0], "Time:", LDensity[1])
print("Lexical Density Amostras:", LDensityA[0], "Time:", LDensityA[1])

print("--------------------------------------------")

LDiversity = LexicalDiversity(text)
LDiversityA = LexicalDiversityA(Samples[0])

print("Lexical Diversity:", LDiversity[0], "Time:", LDiversity[1])
print("Lexical Diversity Amostras:", LDiversityA[0], "Time:", LDiversityA[1])

print("--------------------------------------------")

#print("TTR:", TTRMedio(Samples[0]))
