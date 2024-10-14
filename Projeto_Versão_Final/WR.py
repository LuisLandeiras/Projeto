import AuxFun, spacy, csv, time
import pandas as pd

nlp = spacy.load("en_core_web_sm")

Amostras = AuxFun.Amostras(AuxFun.File("TNasa.txt"))

def Normalize(Valor, Max, Min):
    return (Valor - Min) / (Max - Min)

def CheckWord(input_csv, Word):
    Normalizacao = 0
    
    with open(input_csv, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if Word in row.values():
                Normalizacao = Normalize(int(row['number']), 6281002, 1)
    
    return Normalizacao

def WordRarity(Amostras):
    t = time.process_time()
    Soma = 0
    Words = 0
    for Amostra in Amostras:
        doc = nlp(Amostra)
        Words += len(str(Amostras).split())
        for token in doc:
            if token.is_alpha:
                Soma += CheckWord("BD_Words_Count.csv",token.text.lower())
    
    return Soma/Words, time.process_time() - t

print(WordRarity(Amostras))

#print(CheckWord("BD_Words_Count.csv","the"))