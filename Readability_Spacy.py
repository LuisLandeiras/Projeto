import spacy
from spacy_readability import Readability

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(Readability())

def Readability_Spacy(File): 
    with open(File, "r", encoding='utf-8', errors='ignore') as file:
        sentence = file.read().replace("\n", " ")
    
    doc = nlp(sentence)

    #Ano escolar tendo em conta a compelxidade do texto
    print(doc._.flesch_kincaid_grade_level) 
    print(doc._.flesch_kincaid_reading_ease)
    print(doc._.dale_chall)
    print(doc._.smog)
    print(doc._.coleman_liau_index)
    print(doc._.automated_readability_index)
    print(doc._.forcast)
    print("----------------------------------------------")
    
Readability_Spacy("Textos/biden.txt")
Readability_Spacy("Textos/trump.txt")
Readability_Spacy("Textos/obama.txt")
Readability_Spacy("Textos/The_Mother.txt")
Readability_Spacy("Textos/Men_Without_Women.txt")