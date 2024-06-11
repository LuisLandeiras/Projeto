import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from xgboost import plot_importance
import warnings,AuxFun
import seaborn as sns

def XGBoostTrain(data):
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    data = pd.read_csv("DataV2.csv")
    
    X = data.drop(columns=['Text', 'Classification'])
    y = data['Classification']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if(round(accuracy * 100,2) > 90): model.save_model('XGBModelS.txt')

    print(round(accuracy * 100,2))

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
        'SentimentNeg': Resultado['Sentiment'][0],
        'SentimentNeu': Resultado['Sentiment'][1],
        'SentimentPos': Resultado['Sentiment'][2],
        'Smog': Resultado['Smog'][0],
        'Tree': Resultado['Tree'][0],
        'WordLength': Resultado['WordLength'][0],
    }])

def XGBoostPredict(Texto):
    model = xgb.XGBClassifier()
    model.load_model('XGBModelS.txt')
    
    #data = pd.read_csv("DataV2.csv")
    #feature_names = data.drop(columns=['Text', 'Classification','SentimentNeg','SentimentNeu','SentimentPos']).columns
    #plot_importance(model, importance_type='gain', max_num_features=10)
    #plt.gca().set_yticklabels(feature_names) 
    #plt.show()
    
    #cm = confusion_matrix(y_test, y_pred)
    #plt.figure(figsize=(10, 6))
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    #plt.xlabel('Predicted Label')
    #plt.ylabel('True Label')
    #plt.title('Confusion Matrix')
    #plt.show()
    
    TextoP = preprocess_text(Texto)
    
    sentiment_dict = {'SentimentNeg': TextoP['SentimentNeg'].iloc[0],'SentimentNeu': TextoP['SentimentNeu'].iloc[0],'SentimentPos': TextoP['SentimentPos'].iloc[0]}
    
    SentName = max(sentiment_dict, key=sentiment_dict.get)

    if SentName == 'SentimentNeu': SentimentName = "Neutro"
    if SentName == 'SentimentPos': SentimentName = "Positivo"
    if SentName == 'SentimentNeg': SentimentName = "Negativo"
    
    SentimentValue = sentiment_dict[SentName]
    
    prediction = model.predict(TextoP)
    
    return prediction[0], SentimentName, SentimentValue

#Accuracy 95.83
#Accuracy Sent 91.67

file = AuxFun.File("Texto3_2.txt")
for _ in range(100):
   print(XGBoostPredict(file))

#for _ in range(10):
#    XGBoostTrain("DataV2.csv")