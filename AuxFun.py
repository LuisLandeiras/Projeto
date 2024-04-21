import spacy, random

nlp = spacy.load("en_core_web_sm")

def File(File):
    with open(File, "r", encoding='utf-8', errors='ignore') as file:
        sentence = file.read()
    return sentence

def Amostras(Texto):
    doc = nlp(Texto.lower())

    Palavras = [token.text for token in doc if token.is_alpha] # Separa cada token, guardando só palavras 
    
    paragraphs = []
    current_paragraph = ""
    sentences_count = 0

    # Separa cada paragrafo 
    for sent in doc.sents:
        current_paragraph += sent.text + " "
        sentences_count += 1
        if sentences_count == 10:
            paragraphs.append(current_paragraph.strip())
            current_paragraph = ""
            sentences_count = 0

    # Add the last paragraph if it's not complete
    if current_paragraph:
        paragraphs.append(current_paragraph.strip())
    
    Paragrafos = random.sample(paragraphs,10) 

    #Lista onde é guardada 100 repetições com amostras de 100 palavras da lista Palavras
    Samples = []
    for _ in range(100):
        Sample = random.sample(Palavras,100)
        Samples.append(Sample)
    
    return Samples, Paragrafos # [0] Escolhe de forma random 100 amostras de 100 palavras de uma lista tokenizada; [1] Escolhe de forma random 10 paragrafos de um texto



