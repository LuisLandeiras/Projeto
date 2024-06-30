import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import warnings, AuxFun

def LinearRegressionTrain(FileIn, FileOut, Aux):
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    data = pd.read_csv(FileIn)
    
    if Aux == 'Texto':
        X = data.drop(columns=['Text', 'Classification', 'SentimentNeg', 'SentimentNeu', 'SentimentPos', 'Compound', 'ClassificationS'])
        y = data['Classification']
    elif Aux == 'Sentimento':
        X = data.drop(columns=['Text', 'Classification', 'ClassificationS', 'ARI', 'Coleman', 'Grade', 'LexicalDensity', 'LexicalDiversity', 'Reading', 'SentenceLength', 'Smog', 'Tree', 'WordLength'])
        y = data['ClassificationS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    if Aux == 'Texto':
        if round(r2 * 100, 2) > 90: 
            with open(FileOut, 'wb') as f:
                pickle.dump(model, f)
    elif Aux == 'Sentimento':
        if round(r2 * 100, 2) > 90: 
            with open(FileOut, 'wb') as f:
                pickle.dump(model, f)

    #print(f'Mean Squared Error: {mse}')
    return round(r2 * 100, 2)

def preprocess_text(text):
    Amostra = AuxFun.Amostras(text)
    Resultado = AuxFun.Resultados(Amostra)
    
    return pd.DataFrame([{
        'ARI': Resultado['ARI'][0], 
        'Coleman': Resultado['Coleman'][0], 
        'Grade': Resultado['Grade'][0], 
        'LexicalDensity': Resultado['LexicalDensity'][0],
        'LexicalDiversity': Resultado['LexicalDiversity'][0],
        'Reading': Resultado['Reading'][0],
        'SentenceLength': Resultado['SentenceLength'][0],
        'Smog': Resultado['Smog'][0],
        'Tree': Resultado['Tree'][0],
        'WordLength': Resultado['WordLength'][0],
    }]), pd.DataFrame([{
        'SentimentNeg': Resultado['Sentiment'][0],
        'SentimentNeu': Resultado['Sentiment'][1],
        'SentimentPos': Resultado['Sentiment'][2],
        'Compound': Resultado['Sentiment'][3],
    }])

def LinearRegressionPredict(Texto, Model):
    with open(Model, 'rb') as f:
        model = pickle.load(f)
    
    data = pd.read_csv("DataV4_2_Spacy.csv")
    feature_names = data.drop(columns=['Text', 'Classification', 'SentimentNeg', 'SentimentNeu', 'SentimentPos', 'Compound', 'ClassificationS']).columns

    TextoP = preprocess_text(Texto)[0]
    prediction = model.predict(TextoP)

    return prediction[0], feature_names

def LinearRegressionPredictS(Texto, Model):
    with open(Model, 'rb') as f:
        model = pickle.load(f)
    
    TextoP = preprocess_text(Texto)[1]
    prediction = model.predict(TextoP)

    return prediction[0]

Soma = 0
for _ in range(10):
    for _ in range(100):
        Texto = LinearRegressionTrain('DataV5_Spacy.csv','LinearRegV4_2_Spacy.txt','Texto')
        #print(Texto)
        Soma += Texto
    print(round(Soma/100,2))
    Soma= 0