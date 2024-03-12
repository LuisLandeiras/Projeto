from flask import Flask, render_template, request
import spacy
import nltk
from spacytextblob.spacytextblob import SpacyTextBlob

app = Flask(__name__)

@app.route("/", methods = ["GET","POST"])
def main():
    if request.method == "POST":
        text = request.form.get("text")
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe('spacytextblob')
        doc = nlp(text)
        for token in doc:
            print(token, ": ", token.pos_)
            
        cenas = f"Polaridade: {doc._.blob.polarity} Subjetividade: {doc._.blob.subjectivity}  {doc._.blob.sentiment_assessments.assessments}"
        return render_template('index.html', text=text, cenas=cenas)
    return render_template('index.html')