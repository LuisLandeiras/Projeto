from flask import Flask, render_template, request
import XGBoost, AuxFun

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/sub', methods=['POST'])
def sub():
    result = request.form
    Texto = result.get('text')
    
    Result = Resultados(Texto,'XGBModelV9.txt','XGBModelV9S.txt')
    
    return render_template('sub.html', Result=Result[0], ResultS=Result[1], Algo=Result[2], AlgoRes=Result[3])

@app.route('/texto1', methods=['POST'])
def texto1():
    result = request.form
    Texto = AuxFun.File(result.get('text1'))
    
    Result = Resultados(Texto,'XGBModelV9.txt','XGBModelV9S.txt')
    
    return render_template('texto1.html', Result=Result[0], ResultS=Result[1], Algo=Result[2], Text=Texto, AlgoRes=Result[3])

@app.route('/texto2', methods=['POST'])
def texto2():
    result = request.form
    Texto = AuxFun.File(result.get('text2'))
    
    Result = Resultados(Texto,'XGBModelV9.txt','XGBModelV9S.txt')
    
    return render_template('texto2.html', Result=Result[0], ResultS=Result[1], Algo=Result[2], Text=Texto, AlgoRes=Result[3])

@app.route('/texto3', methods=['POST'])
def texto3():
    result = request.form
    Texto = AuxFun.File(result.get('text3'))
    
    Result = Resultados(Texto,'XGBModelV9.txt','XGBModelV9S.txt')
    
    return render_template('texto3.html', Result=Result[0], ResultS=Result[1], Algo=Result[2], Text=Texto, AlgoRes=Result[3])

def Resultados(Texto, Modelo, ModeloS):

    Resultado = XGBoost.XGBoostPredict(Texto,Modelo)
    ResultadoS = XGBoost.XGBoostPredictS(Texto,ModeloS)
    
    ResRound = [(x, round(y,3)) for x, y in Resultado[1]]
    
    Res = ""
    match Resultado[0]:
        case 0:
            Res = "Estrutura Muito Simples"
        case 1:
            Res = "Estrutura Pouco Complexa"
        case 2:
            Res = "Estrutura Complexa"
        case 3:
            Res = "Estrutura Muito Complexa"

    ResS = ""
    match ResultadoS:
        case 0:
            ResS = "Negativo"
        case 1:
            ResS = "Neutro"
        case 2:
            ResS = "Positivo"

    return Res, ResS, ResRound, Resultado[2]
