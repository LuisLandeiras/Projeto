import spacy, time

nlp = spacy.load("en_core_web_sm")

def average_depth(token, depth=0, count=0):
    if not list(token.children):
        return depth / max(1, count)  # Ensure count is not zero
    else:
        total_depth = sum(average_depth(child, depth + 1, count + 1) for child in token.children)
        return total_depth

def DepthAve(text):
    t = time.process_time()
    doc = nlp(text)
    
    total_depth = sum(average_depth(sent.root) for sent in doc.sents)
    return total_depth / max(1, len(list(doc.sents))), time.process_time() - t

def DepthAveA(samples):
    t = time.process_time()
    total_depth = 0
    total_samples = 0
    for sample in samples:
        doc = nlp(sample)
        for sent in doc.sents:
            total_depth += average_depth(sent.root)
            total_samples += 1
    return total_depth / total_samples, time.process_time() - t