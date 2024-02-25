import spacy
from spacy import displacy

nlp = spacy.load("pt_core_news_lg")
sentence = 'Agradeço o interesse demonstrado por este projeto. É, de facto, um tema muito interessante, com o qual tenho trabalhado há algum tempo. Da minha parte fica já confirmado que serei o seu orientador neste projeto. Podemos assim agendar já uma reunião para esta semana. Proponho quinta-feira às 11h10. Consegue a esta hora?'
doc = nlp(sentence)
for token in doc:
    print(token, "-->", token.pos_)

print(doc.cats)

displacy.serve(doc, style="dep")