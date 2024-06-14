from flask import Flask, render_template, request
import XGBoost

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def sub():
    result = request.form
    Texto = result.get('text')
    
    Resultado = XGBoost.XGBoostPredict(Texto)
    ResultadoS = XGBoost.XGBoostPredictS(Texto)
    return render_template('index.html', Result=Resultado, ResultS=ResultadoS)
