import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import eng_spacysentiment

nlp = eng_spacysentiment.load()

def Sentiment(File):
    with open(File, "r", encoding='utf-8', errors='ignore') as file:
            sentence = file.read().replace("\n", " ")

    doc = nlp(sentence)
    
    #doc._.blob.polarity
    #doc._.blob.subjectivity
    #doc._.blob.sentiment_assessments.assessments
    #doc._.blob.ngrams()
    
    return doc.cats

print(Sentiment("Textos/The_Mother.txt"))
    