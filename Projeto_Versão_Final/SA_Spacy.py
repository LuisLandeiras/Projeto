import spacy, time, asent

def Sentiment(file):
    nlp = spacy.blank('en')
    nlp.add_pipe('sentencizer')
    nlp.add_pipe('asent_en_v1')

    doc = nlp(file)
    Sent = doc._.polarity

    return Sent.negative, Sent.neutral, Sent.positive, Sent.compound
    
def SentimentA(Samples):
    nlp = spacy.blank('en')
    nlp.add_pipe('sentencizer')
    nlp.add_pipe('asent_en_v1')
    
    t = time.process_time()
    neg = 0
    neu = 0
    pos = 0
    compound = 0

    for Sample in Samples:
        doc = nlp(str(Sample))

        Sent = doc._.polarity
        neg += Sent.negative
        neu += Sent.neutral
        pos += Sent.positive
        compound += Sent.compound
                
    return round(neg/len(Samples),3), round(neu/len(Samples),3), round(pos/len(Samples),3), round(compound/len(Samples),3), time.process_time() - t
