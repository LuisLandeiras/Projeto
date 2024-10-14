import spacy
import pandas as pd
import time, AuxFun

nlp = spacy.load("en_core_web_sm")

def Normalize(Valor, Max, Min):
    return (Valor - Min) / (Max - Min)

def LoadBD(csv_file):
    df = pd.read_csv(csv_file)
    word_cache = {}
    
    for _, row in df.iterrows():
        word = row.get("word")
        number = row.get("number")
        
        if isinstance(word, str) and pd.notna(number):
            word_cache[word.lower()] = Normalize(float(number), 6281002, 1)
    
    return word_cache

def CheckWord(word_cache, word):
    word = word.lower()
    if word in word_cache:
        return word_cache[word]
    return 0

def WordRarity(Amostras, word_cache):
    t = time.process_time()
    Soma = 0
    Words = 0
    for Amostra in Amostras:
        doc = nlp(Amostra)
        Words += len(str(Amostra).split())
        for token in doc:
            if token.is_alpha:
                Soma += CheckWord(word_cache, token.text.lower())
    return Soma/Words, time.process_time() - t

Amostras = AuxFun.Amostras(AuxFun.File("TNasa.txt"))
word_cache = LoadBD("BD_Words_Count.csv")

print(WordRarity(Amostras, word_cache))
