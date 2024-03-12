import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def identify_errors(text):
    errors = []
    tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)

    for token, tag in tagged_tokens:
        if tag.startswith('NN') or tag.startswith('VB') or tag.startswith('JJ'):
            synsets = wordnet.synsets(token)
            if not synsets:
                errors.append((token, tag))
    return errors

def correct_errors(text, errors):
    corrected_text = text
    for error in errors:
        token, _ = error
        suggestions = wordnet.synsets(token)
        if suggestions:
            suggested_word = suggestions[0].lemmas()[0].name()
            corrected_text = corrected_text.replace(token, suggested_word)
    return corrected_text

with open("trump.txt", "r", encoding='utf-8') as file:
    sentence = file.read().replace("\n", " ")
    
print(identify_errors(sentence))
print(correct_errors(sentence,identify_errors(sentence)))      
