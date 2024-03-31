#Text Summarization
import nltk, time, AuxFun
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from heapq import nlargest

def TextSummarization(text):
    t = time.process_time()
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Tokenize the sentences into words and remove stopwords
    stop_words = set(stopwords.words('english'))
    word_freq = FreqDist(word.lower() for word in word_tokenize(text) if word.isalnum() and word.lower() not in stop_words)

    # Calculate the score of each sentence based on the frequency of its words
    sentence_scores = {sentence: sum(word_freq[word.lower()] for word in word_tokenize(sentence) if word.isalnum()) for sentence in sentences}

    # Get the top N sentences with the highest scores
    summary_sentences = nlargest(2, sentence_scores, key=sentence_scores.get)

    # Combine the top sentences to form the summary
    summary = ' '.join(summary_sentences)
    return summary, time.process_time() - t 

#text = AuxFun.File("Textos/The_Mother.txt")
#
#print(TextSummarization(text))
