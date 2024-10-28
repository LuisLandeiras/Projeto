import spacy, time, warnings
from transformers import pipeline

nlp = spacy.load("en_core_web_lg")

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=3)

DictEmotion = {
    'anger' : 0.0,
    'disgust': 0.0,
    'fear': 0.0,
    'joy': 0.0,
    'neutral': 0.0,
    'sadness': 0.0,
    'surprise': 0.0
}

def Emotions(Amostras):
    t = time.process_time()
    for Amostra in Amostras:
        emotion_scores = emotion_classifier(Amostra)
        for emotion in emotion_scores[0]:
            if emotion['label'] in DictEmotion:
                DictEmotion[emotion['label']] += round(emotion['score'],3)
    return DictEmotion, time.process_time() - t

#Pensar como colocar para ser usado com teste