from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import AuxFun

Texto = AuxFun.File("Textos/obama.txt")
Texto1 = AuxFun.File("Textos/trump.txt")
Texto2 = AuxFun.File("Textos/biden.txt")
Texto3 = AuxFun.File("Textos/The_Mother.txt")
Texto4 = AuxFun.File("Textos/Men_Without_Women.txt")

Lista = []
Lista.append(Texto)
Lista.append(Texto1)
Lista.append(Texto2)
Lista.append(Texto3)
Lista.append(Texto4)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(Lista)

# NMF decomposition
num_topics = 5
nmf_model = NMF(n_components=num_topics)
nmf_matrix = nmf_model.fit_transform(tfidf_matrix)

# Print the topics and associated words
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(nmf_model.components_):
    print(f"Topic {topic_idx + 1}:")
    top_words_idx = topic.argsort()[-10:][::-1]  # Top 10 words per topic
    top_words = [feature_names[i] for i in top_words_idx]
    print(top_words)
    

