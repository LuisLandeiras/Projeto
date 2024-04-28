# Readability Metrics
import spacy, syllapy, math, re, time
from spacy_readability import Readability

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(Readability())

def Read(File): 
    t = time.process_time()
    doc = nlp(File)

    fleschg = doc._.flesch_kincaid_grade_level
    fleschr = doc._.flesch_kincaid_reading_ease
    ari = doc._.automated_readability_index
    coleman = doc._.coleman_liau_index
    smog = doc._.smog
    
    return fleschg, fleschr, ari, coleman, smog, time.process_time() - t

#0-50
#51-100
#101-150
#151-200
#201-300
#301-500
def SMOGA(Samples):
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        doc = nlp(Sample)
        
        sentences = len(list(doc.sents)) # Conta o número de frases
        
        ComplexWords = sum(1 for word in Sample.split() if syllapy.count(word) >= 3) # Conta o numero de palavras com mais de 3 silabas
        
        Soma += 1.043 * math.sqrt((ComplexWords * (30 / sentences))) + 3.1291 # Formula para calcular SMOG
    return Soma/len(Samples), time.process_time() - t

#0-1.9  Preschool
#2.0-3.9  Kindergarten-1st Grade
#4.0-5.9  2nd-3rd Grade
#6.0-7.9  4th-5th Grade
#8.0-9.9  6th-7th Grade
#10.0-11.9  8th-9th Grade
#12.0-13.9  10th-12th Grade (High School)
#14.0-15.9  College
#16.0+  College Graduate or Beyond
def ColemanA(Samples):
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        doc = nlp(Sample.lower())
        
        Palavras = [token.text for token in doc if token.is_alpha] # Lista com todas as Palavras do Paragrafo
        words = len(Palavras) # Numero de Palavras
        
        sentences = len(list(doc.sents)) # Conta o número de frases
        
        characters = len(re.sub(r'[^a-zA-Z\s]+|\s+', '', Sample)) # Retira os espaços para contar o número de letras usadas
        
        L = (characters/words) * 100
        S = (sentences/words) * 100
        Soma += 0.0588 * L - 0.296 * S - 15.8 # Formula para calcular Coleman
    return Soma/len(Samples), time.process_time() - t

#0-5
#6-8
#9-12
#13-16: College level and above
def FleschGradeA(Samples):
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        doc = nlp(Sample.lower())
        
        Palavras = [token.text for token in doc if token.is_alpha] # Lista com todas as Palavras do Paragrafo
        words = len(Palavras) # Numero de Palavras
        
        syllables = sum(syllapy.count(word) for word in Sample.split())
        
        sentences = len(list(doc.sents)) # Conta o número de frases
        
        Soma += 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59 # Formula para calcular Flesch Grade
    return Soma/len(Samples), time.process_time() - t

#90-100
#80-89
#70-79
#60-69
#50-59
#30-49
#0-29
def FleschReadingA(Samples):
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        doc = nlp(Sample.lower())
        
        Palavras = [token.text for token in doc if token.is_alpha] # Lista com todas as Palavras do Paragrafo
        words = len(Palavras) # Numero de Palavras
        
        syllables = sum(syllapy.count(word) for word in Sample.split())
        
        sentences = len(list(doc.sents)) # Conta o número de frases
        
        Soma += 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words) # Formula para calcular Flesch Reading
    return Soma/len(Samples), time.process_time() - t

def ARIA(Samples):
    t = time.process_time()
    ari = 0
    for Sample in Samples:
        doc = nlp(Sample.lower())
        
        Palavras = [token.text for token in doc if token.is_alpha] # Lista com todas as Palavras do Paragrafo
        words = len(Palavras) # Numero de Palavras
        
        sentences = len(list(doc.sents)) # Conta o número de frases

        characters = len(re.sub(r'[^a-zA-Z\s]+|\s+', '', Sample)) # Retira os espaços para contar o número de letras usadas

        ari += 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43 # Formula para calcular o ARI
    return ari/len(Samples), time.process_time() - t
