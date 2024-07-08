import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from xgboost import plot_importance
import warnings
import AuxFun, os
import seaborn as sns

def XGBoostTrain(FileIn, FileOut, Aux):
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    data = pd.read_csv(FileIn)
    
    if Aux == 'Texto':
        X = data.drop(columns=['Text', 'Classification','SentimentNeg','SentimentNeu','SentimentPos','Compound','ClassificationS'])
        y = data['Classification']
    elif Aux == 'Sentimento':
        X = data.drop(columns=['Text', 'Classification','ClassificationS','ARI','Coleman','Grade','LexicalDensity','LexicalDiversity','Reading','SentenceLength','Smog','Tree','WordLength'])
        y = data['ClassificationS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    
    if Aux == 'Texto':
        if round(accuracy * 100,2) > 98: model.save_model(FileOut)
    elif Aux == 'Sentimento':
        if round(accuracy * 100,2) > 90: model.save_model(FileOut)

    #cm = confusion_matrix(y_test, y_pred)
    #plt.figure(figsize=(10, 6))
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    #plt.xlabel('Predicted Label')
    #plt.ylabel('True Label')
    #plt.title('Confusion Matrix')
    #plt.show()
    
    #data = pd.read_csv("DataV9_Spacy.csv")
    #feature_names = data.drop(columns=['Text', 'Classification','SentimentNeg','SentimentNeu','SentimentPos','Compound','ClassificationS']).columns
    #plot_importance(model, importance_type='gain', max_num_features=10)
    #plt.gca() 
    #plt.show()
    
    return round(accuracy * 100,2), round(precision * 100,2), round(recall * 100,2), round(f1 * 100,2)

def preprocess_text(text):
    Amostra = AuxFun.Amostras(text)
    Resultado = AuxFun.Resultados(Amostra)
    
    return pd.DataFrame([{
        'ARI': Resultado['ARI'][1], 
        'Coleman': Resultado['Coleman'][1], 
        'Grade': Resultado['Grade'][1], 
        'LexicalDensity': Resultado['LexicalDensity'][0],
        'LexicalDiversity': Resultado['LexicalDiversity'][0],
        'Reading': Resultado['Reading'][1],
        'SentenceLength': Resultado['SentenceLength'][1],
        'Smog': Resultado['Smog'][1],
        'Tree': Resultado['Tree'][1],
        'WordLength': Resultado['WordLength'][1],
    }]), pd.DataFrame([{
        'SentimentNeg': Resultado['Sentiment'][0],
        'SentimentNeu': Resultado['Sentiment'][1],
        'SentimentPos': Resultado['Sentiment'][2],
        'Compound': Resultado['Sentiment'][3],
    }])

def XGBoostPredict(Texto, Model):
    model = xgb.XGBClassifier()
    model.load_model(Model)
    
    data = pd.read_csv("DataV9_Spacy.csv")
    feature_names = data.drop(columns=['Text', 'Classification','SentimentNeg','SentimentNeu','SentimentPos','Compound','ClassificationS']).columns

    booster = model.get_booster()
    importance = booster.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    sorted_features = [(feature_names[int(k[1:])], v) for k, v in sorted_importance]
    
    #plot_importance(model, importance_type='gain', max_num_features=10)
    #plt.gca().set_yticklabels([feature_names[int(k[1:])] for k, v in sorted_importance[:10]]) 
    #plt.show()
    
    TextoP = preprocess_text(Texto)[0]
    prediction = model.predict(TextoP)

    return prediction[0], sorted_features, TextoP

def XGBoostPredictS(Texto, Model):
    model = xgb.XGBClassifier()
    model.load_model(Model)
    
    TextoP = preprocess_text(Texto)[1]
    prediction = model.predict(TextoP)

    return prediction[0]
