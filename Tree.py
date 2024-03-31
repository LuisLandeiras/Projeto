import spacy, time, AuxFun

nlp = spacy.load("en_core_web_sm")

# Procurar melhor maneira de pesquisar a arvore(Amostras é muito impreciso)

def MaxDepth(token, depth=1):
    if not list(token.children):
        return depth
    else:
        return max(MaxDepth(child, depth + 1) for child in token.children)

def Depth(Sample):
    t = time.process_time()
    doc = nlp(Sample)
    
    max_depth = max(MaxDepth(sent.root) for sent in doc.sents)
    return max_depth, time.process_time() - t

def DepthA(Samples):
    t = time.process_time()
    Depth = 0
    for Sample in Samples:
        doc = nlp(Sample)
        max_depth = max(MaxDepth(sent.root) for sent in doc.sents)
        if max_depth > Depth:
            Depth = max_depth
    return Depth, time.process_time() - t
        
#text = AuxFun.File("Textos/The_Mother.txt")
#amostra = AuxFun.Amostras(text,40)
#
#BestD = DepthA(amostra[1])
#
#print("Best Depth:", Depth(text)[0], "Time:", Depth(text)[1])
#print("Best Depth Amostra:", BestD[0], "Time:", BestD[1])
#
#print("--------------------------------------------")
#
#text = AuxFun.File("Textos/obama.txt")
#amostra = AuxFun.Amostras(text,15)
#
#BestD = DepthA(amostra[1])
#
#print("Best Depth:", Depth(text)[0], "Time:", Depth(text)[1])
#print("Best Depth Amostra:", BestD[0], "Time:", BestD[1])
#
#print("--------------------------------------------")
#
#text = AuxFun.File("Textos/biden.txt")
#amostra = AuxFun.Amostras(text,15)
#
#BestD = DepthA(amostra[1])
#
#print("Best Depth:", Depth(text)[0], "Time:", Depth(text)[1])
#print("Best Depth Amostra:", BestD[0], "Time:", BestD[1])
#
#print("--------------------------------------------")
#
#text = AuxFun.File("Textos/Men_Without_Women.txt")
#amostra = AuxFun.Amostras(text,40)
#
#BestD = DepthA(amostra[1])
#
#print("Best Depth:", Depth(text)[0], "Time:", Depth(text)[1])
#print("Best Depth Amostra:", BestD[0], "Time:", BestD[1])

