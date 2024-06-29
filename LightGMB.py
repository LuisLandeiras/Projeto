import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, AuxFun

def LightGBMTrain(FileIn, FileOut, Aux):
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    data = pd.read_csv(FileIn)
    
    if Aux == 'Texto':
        X = data.drop(columns=['Text', 'Classification', 'SentimentNeg', 'SentimentNeu', 'SentimentPos', 'Compound', 'ClassificationS'])
        y = data['Classification']
    elif Aux == 'Sentimento':
        X = data.drop(columns=['Text', 'Classification', 'ClassificationS', 'ARI', 'Coleman', 'Grade', 'LexicalDensity', 'LexicalDiversity', 'Reading', 'SentenceLength', 'Smog', 'Tree', 'WordLength'])
        y = data['ClassificationS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = lgb.LGBMClassifier(verbosity=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if Aux == 'Texto':
        if round(accuracy * 100, 2) > 90: model.booster_.save_model(FileOut)
    elif Aux == 'Sentimento':
        if round(accuracy * 100, 2) > 90: model.booster_.save_model(FileOut)

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

def LightGBMPredict(Texto, Model):
    model = lgb.Booster(model_file=Model)
    
    data = pd.read_csv("DataV4_2_Spacy.csv")
    feature_names = data.drop(columns=['Text', 'Classification', 'SentimentNeg', 'SentimentNeu', 'SentimentPos', 'Compound', 'ClassificationS']).columns

    importance = model.feature_importance(importance_type='gain')
    sorted_indices = importance.argsort()[::-1]
    sorted_features = [(feature_names[i], importance[i]) for i in sorted_indices[:10]]
    
    TextoP = preprocess_text(Texto)[0]
    TextoP = lgb.Dataset(TextoP, free_raw_data=False)
    prediction = model.predict(TextoP.data, num_iteration=model.best_iteration)

    return prediction[0].argmax(), sorted_features

def LightGBMPredictS(Texto, Model):
    model = lgb.Booster(model_file=Model)
    
    TextoP = preprocess_text(Texto)[1]
    TextoP = lgb.Dataset(TextoP, free_raw_data=False)
    prediction = model.predict(TextoP.data, num_iteration=model.best_iteration)

    return prediction[0].argmax()


#file = AuxFun.File("Textos_Teste/Sad.txt")
#for _ in range(10):
#    print("NLTK:",LightGBMPredict(file,'XGBModelV4_NLTK.txt'))
#    print("NLTKS:",LightGBMPredictS(file,'XGBModelV4S_2_NLTK.txt'))
#    print("Spacy:",LightGBMPredict(file,'XGBModelV4_Spacy.txt'))
#    print("SpacyS:",LightGBMPredictS(file,'XGBModelV4S_2_Spacy.txt'))
#    print("--------------------------------")

Soma = 0
for _ in range(100):
    Soma += LightGBMTrain('DataV4_2_Spacy.csv','LightGMBV4_2_Spacy.txt','Texto')
print(round(Soma/100,2))