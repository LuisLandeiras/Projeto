import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

nltk.download('all')

#Amostras em paragrafos/numero de palavras maximo

def Sentiment(File) -> float:
    with open(File, "r", encoding='utf-8', errors='ignore') as file:
        sentence = file.read().replace("\n", " ")
    
    tokens = word_tokenize(sentence.lower())
    
    analyzer = SentimentIntensityAnalyzer()
    
    scores = analyzer.polarity_scores(tokens)
    
    return scores

print(Sentiment("Textos/biden.txt"))