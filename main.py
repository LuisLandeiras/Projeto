from flask import Flask, render_template, request
import XGBoost
import pandas as pd

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/sub', methods=['POST'])
def sub():
    result = request.form
    Texto = result.get('text')
    data = pd.read_csv("DataV2.csv")
    
    Resultado = XGBoost.XGBoost(data,Texto)
    
    return render_template('sub.html', Result=Resultado)
