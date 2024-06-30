import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import warnings, Algos.AuxFun as AuxFun

def LogisticRegressionTrain(FileIn, FileOut, Aux):
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    data = pd.read_csv(FileIn)
    
    if Aux == 'Texto':
        X = data.drop(columns=['Text', 'Classification', 'SentimentNeg', 'SentimentNeu', 'SentimentPos', 'Compound', 'ClassificationS'])
        y = data['Classification']
    elif Aux == 'Sentimento':
        X = data.drop(columns=['Text', 'Classification', 'ClassificationS', 'ARI', 'Coleman', 'Grade', 'LexicalDensity', 'LexicalDiversity', 'Reading', 'SentenceLength', 'Smog', 'Tree', 'WordLength'])
        y = data['ClassificationS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if Aux == 'Texto':
        if round(accuracy * 100, 2) > 90: 
            # Salva o modelo
            with open(FileOut, 'wb') as f:
                import pickle
                pickle.dump(model, f)
    elif Aux == 'Sentimento':
        if round(accuracy * 100, 2) > 90: 
            # Salva o modelo
            with open(FileOut, 'wb') as f:
                import pickle
                pickle.dump(model, f)

    return round(accuracy * 100, 2)

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

def LogisticRegressionPredict(Texto, Model):
    with open(Model, 'rb') as f:
        import pickle
        model = pickle.load(f)
    
    data = pd.read_csv("DataV4_2_Spacy.csv")
    feature_names = data.drop(columns=['Text', 'Classification', 'SentimentNeg', 'SentimentNeu', 'SentimentPos', 'Compound', 'ClassificationS']).columns

    TextoP = preprocess_text(Texto)[0]
    prediction = model.predict(TextoP)

    return prediction[0], feature_names

def LogisticRegressionPredictS(Texto, Model):
    with open(Model, 'rb') as f:
        import pickle
        model = pickle.load(f)
    
    TextoP = preprocess_text(Texto)[1]
    prediction = model.predict(TextoP)

    return prediction[0]

Soma = 0
for _ in range(100):
    Texto = LogisticRegressionTrain('DataV4_2_Spacy.csv','LogRegV4_2_Spacy.txt','Texto')
    print(Texto)
    Soma += Texto
print(round(Soma/100,2))