import time, nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#nltk.download('vader_lexicon')

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
        Texto = str(Sample)
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(Texto)
        neg += scores['neg']
        neu += scores['neu']
        pos += scores['pos']
        compound += scores['compound']
    
    return '{:.3}'.format(neg/len(Samples)), '{:.3}'.format(neu/len(Samples)), '{:.3}'.format(pos/len(Samples)), '{:.3}'.format(compound/len(Samples)), time.process_time() - t