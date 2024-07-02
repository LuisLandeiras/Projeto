import spacy, time

nlp = spacy.load("en_core_web_sm")

def Normalize(Valor, Max, Min):
    return (Valor - Min) / (Max - Min)

def average_depth(token, depth=0, count=0):
    if not list(token.children):
        return depth/max(1, count)
    else:
        total_depth = sum(average_depth(child, depth + 1, count + 1) for child in token.children)
        return total_depth

def DepthAveA(Samples):
    t = time.process_time()
    Depth = 0
    for Sample in Samples:
        Texto = str(Sample)
        doc = nlp(Texto)
        for sent in doc.sents:
            Depth += average_depth(sent.root)
            
    Normalizacao = Normalize(Depth/len(Samples),30,1)    
    
    return round(Depth/len(Samples),2), round(Normalizacao,3), time.process_time() - t

def DepthAve(text):
    t = time.process_time()
    Texto = str(text)
    doc = nlp(Texto)
    
    total_depth = sum(average_depth(sent.root) for sent in doc.sents)
    
    return round(total_depth/max(1, len(list(doc.sents))),3), time.process_time() - t

