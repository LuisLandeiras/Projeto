import AuxFun
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def Sentiment(Sample):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(Sample)
    return scores

def SentimentA(Samples):
    neg = 0
    neu = 0
    pos = 0
    compound = 0
    for Sample in Samples:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(Sample)
        neg += scores['neg']
        neu += scores['neu']
        pos += scores['pos']
        compound += scores['compound']
    
    return neg/10, neu/10, pos/10, compound/10

Sentence = AuxFun.File("Textos/biden.txt")
Samples = AuxFun.Amostras(Sentence,10)

print(f"Sentiment negative: {SentimentA(Samples[1])[0]:.5f}")
print(f"Sentiment neutral: {SentimentA(Samples[1])[1]:.5f}")
print(f"Sentiment positive: {SentimentA(Samples[1])[2]:.5f}")
print(f"Compound: {SentimentA(Samples[1])[3]:.5f}")

print("----------------------------------------------")
    
print("Sentiment negative:", Sentiment(Sentence))