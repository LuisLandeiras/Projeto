import spacy, random

nlp = spacy.load("en_core_web_sm")

def File(File):
    with open(File, "r", encoding='utf-8', errors='ignore') as file:
        sentence = file.read()
    return sentence

def Amostras(Texto, type):
    doc = nlp(Texto.lower())
    
    Palavras = [token.text for token in doc if token.is_alpha] # Separa cada token, guardando só palavras 
    if type == 1:
        Paragrafos = Texto.split("\r\n\r\n") # Separa cada paragrafo 
    elif type == 2:
        Paragrafos = Texto.split("\n\n") # Separa cada paragrafo 

    Paragrafos = random.sample(Paragrafos, 10)
    
    #Lista onde é guardada 100 repetições com amostras de 100 palavras da lista Palavras
    Samples = []
    for _ in range(100):
        Sample = random.sample(Palavras,100)
        Samples.append(Sample)
    
    return Samples, Paragrafos # [0] Escolhe de forma random 100 amostras de 100 palavras de uma lista tokenizada; [1] Escolhe de forma random 10 paragrafos de um texto