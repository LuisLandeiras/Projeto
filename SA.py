import time, nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def Sentiment(Sample):
    t = time.process_time()
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(Sample)
    return scores, time.process_time() - t

def SentimentA(Samples):
    t = time.process_time()
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
    
    return '{:.3}'.format(neg/10), '{:.3}'.format(neu/10), '{:.3}'.format(pos/10), '{:.3}'.format(compound/10), time.process_time() - t