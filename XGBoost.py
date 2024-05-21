import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance
import warnings

def XGBoost():
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

Soma = 0
for i in range(100):
    Result = XGBoost()
    print(Result)
    Soma += Result

print("Media:", Soma/100)
    
    

