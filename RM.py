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

def SMOGA(Samples):
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        doc = nlp(Sample)
        
        sentences = len(list(doc.sents)) # Conta o número de frases
        
        ComplexWords = sum(1 for word in Sample.split() if syllapy.count(word) >= 3) # Conta o numero de palavras com mais de 3 silabas
        
        Soma += 1.043 * math.sqrt((ComplexWords * (30 / sentences))) + 3.1291 # Formula para calcular SMOG
    return '{:.3}'.format(Soma/len(Samples)), time.process_time() - t

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
    return '{:.3}'.format(Soma/len(Samples)), time.process_time() - t

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
    return '{:.3}'.format(Soma/len(Samples)), time.process_time() - t

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
    return '{:.3}'.format(Soma/len(Samples)), time.process_time() - t

def ARIA(Samples):
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        doc = nlp(Sample.lower())
        
        Palavras = [token.text for token in doc if token.is_alpha] # Lista com todas as Palavras do Paragrafo
        words = len(Palavras) # Numero de Palavras
        
        sentences = len(list(doc.sents)) # Conta o número de frases

        characters = len(re.sub(r'[^a-zA-Z\s]+|\s+', '', Sample)) # Retira os espaços para contar o número de letras usadas

        Soma += 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43 # Formula para calcular o ARI
    return '{:.3}'.format(Soma/len(Samples)), time.process_time() - t
