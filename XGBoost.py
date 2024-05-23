import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance
import warnings,AuxFun

def XGBoostTeste():
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    data = pd.read_csv("DataV2.csv")

    X = data.drop(columns=['Text','Classification'])
    y = data['Classification']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = xgb.XGBClassifier(label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    #plot_importance(model, importance_type='gain', max_num_features=14)
    #plt.show()
    #
    #cm = confusion_matrix(y_test, y_pred)
    #plt.figure(figsize=(10, 6))
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    #plt.xlabel('Predicted Label')
    #plt.ylabel('True Label')
    #plt.title('Confusion Matrix')
    #plt.show()
    
    return round(accuracy*100,2)

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
        #'SentimentNeg': Resultado['Sentiment'][0],
        #'SentimentNeu': Resultado['Sentiment'][1],
        #'SentimentPos': Resultado['Sentiment'][2],
        'Smog': Resultado['Smog'][0],
        'Tree': Resultado['Tree'][0],
        'WordLength': Resultado['WordLength'][0],
    }])

def XGBoost(data, new_text=None):
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    X = data.drop(columns=['Text', 'Classification','SentimentNeg','SentimentNeu','SentimentPos'])
    y = data['Classification']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    p = 0
    
    if new_text:
        new_text_processed = preprocess_text(new_text)
        prediction = model.predict(new_text_processed)
        p = prediction[0]

    return round(accuracy * 100, 2), p

#def main():
#    data = pd.read_csv("DataV2.csv")
#    Soma = 0
#    for i in range(100):
#        new_text = AuxFun.File("Texto.txt")
#        classification = XGBoost(data, new_text)
#        print(f"Iteration {i+1} Classification for new text: {classification[1]}")
#        print(f"Iteration {i+1} Accuracy for new text: {classification[0]}")
#        Soma += classification[0]
#
#    print("Average Accuracy:", Soma / 100)
    

    
    

