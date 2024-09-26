import spacy, AuxFun, time
from transformers import pipeline

t = time.process_time()

# Load spaCy English model
nlp = spacy.load("en_core_web_lg")

# Load a pre-trained emotion classifier from Hugging Face's transformers library
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=3)

Amostra = AuxFun.Amostras(AuxFun.File("TNasa.txt"))

for a in Amostra:
    # Run the emotion classifier on the original text
    emotion_scores = emotion_classifier(a)

    # Display the results
    for emotion in emotion_scores[0]:
        print(f"{a}: {emotion['label']}: {emotion['score']:.2f}")

print(time.process_time() - t)