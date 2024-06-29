from flask import Flask, render_template, request
import Modelos.XGBoost as XGBoost, Algos.AuxFun as AuxFun, Modelos.LightGMB as LightGMB

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/sub', methods=['POST'])
def sub():
    result = request.form
    Texto = result.get('text')
    
    Resultado = XGBoost.XGBoostPredict(Texto,'Treinos/XGBModelV5_Spacy.txt')
    ResultadoS = XGBoost.XGBoostPredictS(Texto,'V4/XGBModelV4S_2_Spacy.txt')
    
    #Resultado = LightGMB.LightGBMPredict(Texto,'Treinos/GMBModelV5_2_Spacy.txt')
    #ResultadoS = LightGMB.LightGBMPredictS(Texto,'Treinos/GMBModelV5S_2_Spacy.txt')
       
    ResRound = [(x, round(y,3)) for x, y in Resultado[1]]
    
    Res = ""
    match Resultado[0]:
        case 0:
            Res = "Muito Mau"
        case 1:
            Res = "Mau"
        case 2:
            Res = "Bom"
        case 3:
            Res = "Muito Bom"

    ResS = ""
    match ResultadoS:
        case 0:
            ResS = "Negativo"
        case 1:
            ResS = "Neutro"
        case 2:
            ResS = "Positivo"
    
    return render_template('sub.html', Result=Res, ResultS=ResS, Algo=ResRound)

@app.route('/texto1', methods=['POST'])
def texto1():
    result = request.form
    Texto = AuxFun.File(result.get('text1'))
    
    Resultado = XGBoost.XGBoostPredict(Texto,'Treinos/XGBModelV5_Spacy.txt')
    ResultadoS = XGBoost.XGBoostPredictS(Texto,'V4/XGBModelV4S_2_Spacy.txt')
    
    ResRound = [(x, round(y,3)) for x, y in Resultado[1]]
    
    Res = ""
    match Resultado[0]:
        case 0:
            Res = "Muito Mau"
        case 1:
            Res = "Mau"
        case 2:
            Res = "Bom"
        case 3:
            Res = "Muito Bom"

    ResS = ""
    match ResultadoS:
        case 0:
            ResS = "Negativo"
        case 1:
            ResS = "Neutro"
        case 2:
            ResS = "Positivo"
    
    return render_template('texto1.html', Result=Res, ResultS=ResS, Algo=ResRound, Text=Texto)

@app.route('/texto2', methods=['POST'])
def texto2():
    result = request.form
    Texto = AuxFun.File(result.get('text2'))
    
    Resultado = XGBoost.XGBoostPredict(Texto,'Treinos/XGBModelV5_Spacy.txt')
    ResultadoS = XGBoost.XGBoostPredictS(Texto,'V4/XGBModelV4S_2_Spacy.txt')
    
    ResRound = [(x, round(y,3)) for x, y in Resultado[1]]
    
    Res = ""
    match Resultado[0]:
        case 0:
            Res = "Muito Mau"
        case 1:
            Res = "Mau"
        case 2:
            Res = "Bom"
        case 3:
            Res = "Muito Bom"

    ResS = ""
    match ResultadoS:
        case 0:
            ResS = "Negativo"
        case 1:
            ResS = "Neutro"
        case 2:
            ResS = "Positivo"
    
    return render_template('texto2.html', Result=Res, ResultS=ResS, Algo=ResRound, Text=Texto)

@app.route('/texto3', methods=['POST'])
def texto3():
    result = request.form
    Texto = AuxFun.File(result.get('text3'))
    
    Resultado = XGBoost.XGBoostPredict(Texto,'Treinos/XGBModelV5_Spacy.txt')
    ResultadoS = XGBoost.XGBoostPredictS(Texto,'V4/XGBModelV4S_2_Spacy.txt')
    
    ResRound = [(x, round(y,3)) for x, y in Resultado[1]]
    
    Res = ""
    match Resultado[0]:
        case 0:
            Res = "Muito Mau"
        case 1:
            Res = "Mau"
        case 2:
            Res = "Bom"
        case 3:
            Res = "Muito Bom"

    ResS = ""
    match ResultadoS:
        case 0:
            ResS = "Negativo"
        case 1:
            ResS = "Neutro"
        case 2:
            ResS = "Positivo"
    
    return render_template('texto3.html', Result=Res, ResultS=ResS, Algo=ResRound, Text=Texto)
