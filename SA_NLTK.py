import random
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def File(File):
    with open(File, "r", encoding='utf-8', errors='ignore') as file:
        content = file.read()
        sentence = content.split('\n\n')
    return sentence

def Sentiment(File):
    neg = 0
    neu = 0
    pos = 0
    compound = 0
    Paragrafo = random.sample(File,10)
    
    for i in range(10):
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(Paragrafo[i])
        neg += scores['neg']
        neu += scores['neu']
        pos += scores['pos']
        compound += scores['compound']
    
    return neg/10, neu/10, pos/10, compound/10

Sentence = File("Textos/biden.txt")

print(f"Sentiment negative: {Sentiment(Sentence)[0]:.5f}")
print(f"Sentiment neutral: {Sentiment(Sentence)[1]:.5f}")
print(f"Sentiment positive: {Sentiment(Sentence)[2]:.5f}")
print(f"Compound: {Sentiment(Sentence)[3]:.5f}")