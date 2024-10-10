import AuxFun, spacy, csv, time
import pandas as pd

nlp = spacy.load("en_core_web_lg")

Amostras = AuxFun.Amostras(AuxFun.File("TNasa.txt"))

def Normalize(Valor, Max, Min):
    return (Valor - Min) / (Max - Min)

def CheckWord(input_csv, Word):
    Normalizacao = 0
    
    df = pd.read_csv(input_csv)
    
    for index, row in df.iterrows():
        if Word in row["word"]:
            Normalizacao = Normalize(row['number'], 6281002, 1)
    
    return Normalizacao

def WordRarity(Amostras):
    t = time.process_time()
    Soma = 0
    for Amostra in Amostras:
        doc = nlp(Amostra)
        for token in doc:
            if token.is_alpha:
                Soma += CheckWord("BD_Words_Count.csv",token.text.lower())
    
    return Soma, time.process_time() - t

#print(WordRarity(Amostras))


print(CheckWord("BD_Words_Count.csv","the"))