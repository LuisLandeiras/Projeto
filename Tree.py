import spacy, time

nlp = spacy.load("en_core_web_sm")

def average_depth(token, depth=0, count=0):
    if not list(token.children):
        return depth/max(1, count)
    else:
        total_depth = sum(average_depth(child, depth + 1, count + 1) for child in token.children)
        return total_depth

def DepthAve(text):
    t = time.process_time()
    Texto = str(text)
    doc = nlp(Texto)
    
    total_depth = sum(average_depth(sent.root) for sent in doc.sents)
    return round(total_depth/max(1, len(list(doc.sents))),3), time.process_time() - t

def DepthAveA(samples):
    t = time.process_time()
    Depth = 0
    for Sample in samples:
        Texto = str(Sample)
        doc = nlp(Texto)
        for sent in doc.sents:
            Depth += average_depth(sent.root)
    return round(Depth/len(samples),2), time.process_time() - t