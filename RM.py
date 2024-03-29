# Readability Metrics
import spacy, AuxFun
from spacy_readability import Readability

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(Readability())

def Flesch(File): 
    doc = nlp(File)

    #Ano escolar tendo em conta a compelxidade do texto
    fleschg = doc._.flesch_kincaid_grade_level
    fleschr = doc._.flesch_kincaid_reading_ease
    #print(doc._.dale_chall)
    #print(doc._.smog)
    #print(doc._.coleman_liau_index)
    #print(doc._.automated_readability_index)
    #print(doc._.forcast)
    
    return fleschg, fleschr

# Fazer á manete(Resultados muito imprecisos)
def FleschA(Samples):
    fleschg = 0
    fleschr = 0
    for Sample in Samples:
        doc = nlp(str(Sample))
        fleschg += doc._.flesch_kincaid_grade_level
        fleschr += doc._.flesch_kincaid_reading_ease
    return fleschg/10, fleschr/10

def ARI(Sample):
    words = len(Sample.split())
    sentences = Sample.count('.') + Sample.count('!') + Sample.count('?') #Conta o número de frases tendo em conta as pontuações
    characters = len(Sample.replace(" ", "")) #Retira os espaços para contar o número de letras usadas
    ari = 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43 #Formula usada para calcular o ARI
    return ari

def ARIA(Samples):
    ari = 0
    for Sample in Samples: 
        words = len(Sample.split())
        sentences = Sample.count('.') + Sample.count('!') + Sample.count('?') #Conta o número de frases tendo em conta as pontuações
        characters = len(Sample.replace(" ", "")) #Retira os espaços para contar o número de letras usadas
        ari += 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43 #Formula usada para calcular o ARI
    return ari/len(Samples)

Text = AuxFun.File("Textos/biden.txt")
Sample = AuxFun.Amostras(Text,10)

print("ARI Amostra:", ARIA(Sample[1]))
print("ARI:", ARI(Text))

print("------------------------------------")

print("Flesch Grade:", Flesch(Text)[0])
print("Flesch Grade Amostra:", FleschA(Sample)[0])

print("------------------------------------")

print("Flesch Reading:", Flesch(Text)[1])
print("Flesch Reading Amostra:", FleschA(Sample)[1])


