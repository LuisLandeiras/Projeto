# Readability Metrics
import spacy, syllapy, math, re, time
import textdescriptives as td

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('textdescriptives/readability')

def Normalize(Valor, Max, Min):
    return (Valor - Min) / (Max - Min)

def Read(File):
    t = time.process_time()
    doc = nlp(File)
    
    ari = doc._.readability['automated_readability_index']
    fleschg = doc._.readability['flesch_kincaid_grade']
    coleman = doc._.readability['coleman_liau_index']
    smog = doc._.readability['smog']
    gfog = doc._.readability['gunning_fog']
    lix = doc._.readability['lix']
    rix = doc._.readability['rix']
        
    return round(fleschg,3), round(ari,3), round(coleman,3), round(smog,3), round(gfog,3), round(lix,3), round(rix,3), time.process_time() - t

def ReadA(Samples):
    t = time.process_time()
    
    ari,fleschg,coleman,smog,gfog,lix,rix = 0,0,0,0,0,0,0
    
    for Sample in Samples:
        doc = nlp(Sample)

        ari += doc._.readability['automated_readability_index']
        fleschg += doc._.readability['flesch_kincaid_grade']
        coleman += doc._.readability['coleman_liau_index']
        gfog += doc._.readability['gunning_fog']
        lix += doc._.readability['lix']
        rix += doc._.readability['rix']
    
    NormalizacaoARI = Normalize(ari/len(Samples),14,0)
    NormalizacaoGrade = Normalize(fleschg/len(Samples),18,0)
    NormalizacaoColeman = Normalize(coleman/len(Samples),12,0)
    NormalizacaoGFog = Normalize(gfog/len(Samples),17,6)
    NormalizacaoLix = Normalize(lix/len(Samples),56,0)
    NormalizacaoRix = Normalize(rix/len(Samples),16,0)
    
    return round(NormalizacaoGrade,3), round(NormalizacaoARI,3), round(NormalizacaoColeman,3), round(NormalizacaoGFog,3), round(NormalizacaoLix,3), round(NormalizacaoRix,3), time.process_time() - t

def SMOGA(Samples):
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        Texto = str(Sample)
        
        ComplexWords = sum(1 for word in Texto.split() if syllapy.count(word) >= 3) # Conta o numero de palavras com mais de 3 silabas
        
        Soma += 1.043 * math.sqrt(ComplexWords * 30) + 3.1291 # Formula para calcular SMOG
        
    Normalizacao = Normalize(Soma/len(Samples),12,0)
    
    return round(Soma/len(Samples),3), round(Normalizacao,3), time.process_time() - t

def ColemanA(Samples):
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        Texto = str(Sample)

        words = len(Texto.split()) # Numero de Palavras
        
        characters = len(re.sub(r'[^a-zA-Z\s]+|\s+', '', Texto)) # Retira os espaços para contar o número de letras usadas
        
        L = (characters/words) * 100
        S = (1/words) * 100
        Soma += 0.0588 * L - 0.296 * S - 15.8 # Formula para calcular Coleman
        
    Normalizacao = Normalize(Soma/len(Samples),12,0)
    
    return round(Soma/len(Samples),3), round(Normalizacao,3), time.process_time() - t

def FleschGradeA(Samples):
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        Texto = str(Sample)

        words = len(Texto.split()) # Numero de Palavras
        
        syllables = sum(syllapy.count(word) for word in Texto.split())
        
        Soma += 0.39 * words + 11.8 * (syllables/words) - 15.59 # Formula para calcular Flesch Grade
    
    Normalizacao = Normalize(Soma/len(Samples),18,0) 
    
    return round(Soma/len(Samples),3), round(Normalizacao,3), time.process_time() - t

def ARIA(Samples):
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        Texto = str(Sample)

        words = len(Texto.split()) # Numero de Palavras

        characters = len(re.sub(r'[^a-zA-Z\s]+|\s+', '', Texto)) # Retira os espaços para contar o número de letras usadas

        Soma += 4.71 * (characters/words) + 0.5 * words - 21.43 # Formula para calcular o ARI
        
    Normalizacao = Normalize(Soma/len(Samples),14,0)
    
    return round(Soma/len(Samples),3), round(Normalizacao,3), time.process_time() - t

def GFogA(Samples):
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        Texto = str(Sample)

        Words = len(Texto.split()) # Numero de Palavras

        ComplexWords = sum(1 for word in Texto.split() if syllapy.count(word) >= 3) # Conta o numero de palavras com mais de 3 silabas
        
        Soma += 0.4 * (Words + (100 * (ComplexWords/Words))) 
        
    Normalizacao = Normalize(Soma/len(Samples),17,6)
    
    return round(Soma/len(Samples),3), round(Normalizacao,3), time.process_time() - t

def LixA(Samples):
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        Texto = str(Sample)
        
        Words = len(Texto.split())
        
        LongWords = sum(1 for word in Texto.split() if len(word) > 6)
        
        Soma += Words + (LongWords * 100) / Words 

    Normalizacao = Normalize(Soma/len(Samples), 56,0)
    
    return round(Soma/len(Samples),3), round(Normalizacao,3), time.process_time() - t

def RixA(Samples):
    t = time.process_time()
    Soma = 0
    for Sample in Samples:
        Texto = str(Sample)
        
        Words = Texto.split()
        
        LongWords = sum(1 for word in Words if len(word) > 6)
        
        Soma += LongWords

    Normalizacao = Normalize(Soma/len(Samples), 16,0)
    
    return round(Soma/len(Samples),3), round(Normalizacao,3), time.process_time() - t
