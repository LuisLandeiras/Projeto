import spacy
from transformers import pipeline

# Load spaCy English model
nlp = spacy.load("en_core_web_lg")

# Load a pre-trained emotion classifier from Hugging Face's transformers library
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=3)

# Sample text
text = "I am so happy and excited today! I went to the park and had fun!"

# Process the text using spaCy
doc = nlp(text)

# Run the emotion classifier on the original text
emotion_scores = emotion_classifier(text)

# Display the results
for emotion in emotion_scores[0]:
    print(f"{emotion['label']}: {emotion['score']:.2f}")
