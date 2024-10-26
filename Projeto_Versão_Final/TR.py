import spacy, AuxFun, time
import gensim
from gensim import corpora

nlp = spacy.load("en_core_web_lg")

def preprocess(text):
    doc = nlp(text)

    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

def TextTopic(Amostras):
    t = time.process_time()
    TopicList = []
    texts = [preprocess(doc) for doc in Amostras]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda_model = gensim.models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

    for idx, topic in lda_model.print_topics(-1):
        TopicList.append(topic)
    
    return TopicList, time.process_time() - t

Amostras = AuxFun.Amostras(AuxFun.File("TNasa.txt"))

print(TextTopic(Amostras))

#Pensar como colocar para ser usado com teste