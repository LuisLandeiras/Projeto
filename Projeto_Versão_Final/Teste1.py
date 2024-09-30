import spacy, AuxFun

nlp = spacy.load("en_core_web_lg")  #"en_core_web_md"

Amostra = AuxFun.File("TNasa.txt")

text = "This is a sample text. This text is for NLP processing."
doc = nlp(Amostra)

for token in doc:
    if token.is_alpha:
        print(f"Word: {token.text}, Probability: {token.prob}")
