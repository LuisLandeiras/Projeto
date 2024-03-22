import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Load the English language model
nlp = spacy.load("en_core_web_sm")

def File(File):
    with open(File, "r", encoding='utf-8', errors='ignore') as file:
        sentence = file.read()
        
    return sentence

# Process the input texts
def TextSimilarity(File1, File2):
    
    text1 = nlp(File1)
    text2 = nlp(File2)

    # Compute vectors for the processed texts
    vector1 = text1.vector.reshape(1, -1)
    vector2 = text2.vector.reshape(1, -1)

    # Compute cosine similarity
    similarity = cosine_similarity(vector1, vector2)[0][0]
    print("Cosine Similarity:", similarity)
    
File1 = File("Textos/obama.txt")
File2 = File("Textos/The_Mother.txt")

TextSimilarity(File1, File2)
