#Text Complexity Measures
import spacy, random, string

nlp = spacy.load('en_core_web_sm')

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

def Amostras(Texto):
    Texto = Texto.split()
    Samples = []
    Texto = random.sample(Texto,1000)
    for _ in range(100): #100 samples são estudadas para o resultado final
        Sample = random.sample(Texto,10) #Escolhe 10 palavras random do texto
        Samples.append(Sample)
    return Samples, Texto      

def LexicalDesnsityA(Samples):
    soma = 0
    for Sample in Samples:
        for Amostra in Sample:
            doc = nlp(Amostra)
            for token in doc:
                if token.pos_ == "ADP" or token.pos_ == "AUX" or token.pos_ == "PART" or token.pos_ == "CCONJ" or token.pos_ == "NOUN" or token.pos_ == "INTJ" or token.pos_ == "PROPN" or token.pos_ == "ADJ":
                    soma += 1
    return soma/1000

def LexicalDiversityA(Samples):
    DiffWords = list(dict.fromkeys(Samples))
    return len(DiffWords)/1000

def WordLengthA(Samples):
    WordLength = sum(len(Sample) for Sample in Samples)
    return WordLength/1000

def TTRMedio(File) -> float:
    with open(File, "r", encoding='utf-8', errors='ignore') as file:
        sentence = file.read().replace("\n", " ")
        sentence = sentence.translate(str.maketrans('','', string.punctuation)) #Limpar a pontuação do texto
    
    Text = sentence.split() #Colocar o texto numa lista

    soma = 0
    for _ in range(1000): #1000 samples são estudadas para o resultado final
        Sample = random.sample(Text,500) #Escolhe 500 palavras random do texto
        ttr = len(set(Sample))/len(Sample) #Divisão entre a sample limpa(Palavras não repetidas) e o tamanho total da sample
        soma += ttr
    return soma/1000

print(f"TTR Biden: {TTRMedio('Textos/biden.txt'):.5f}")
print(f"TTR Trump: {TTRMedio('Textos/trump.txt'):.5f}")
print(f"TTR Obama: {TTRMedio('Textos/obama.txt'):.5f}")

print(f"TTR The_Mother: {TTRMedio('Textos/The_Mother.txt'):.5f}")
print(f"TTR Men_Withour_Women: {TTRMedio('Textos/Men_Without_Women.txt'):.5f}")

text = File("Textos/The_Mother.txt")

print("Average Sentence Length:", SentenceLength(text))

print("--------------------------------------------")

print("Average Word Length:", WordLength(text))
print("Average Word Length:", WordLengthA(Amostras(text)[1]))

print("--------------------------------------------")

print("Lexical Density:", LexicalDensity(text))
print("Lexical Density Amostras:", LexicalDesnsityA(Amostras(text)[0]))

print("--------------------------------------------")

print("Lexical Diversity:", LexicalDiversity(text))
print("Lexical Diversity Amostras:", LexicalDiversityA(Amostras(text)[1]))