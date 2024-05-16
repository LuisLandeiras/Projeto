import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance

def XGBoost():
    data = pd.read_csv("Data.csv")

    X = data.drop(columns=['Text','Classification'])
    y = data['Classification']

    #d = pd.DataFrame(X)
    #d['y'] = y

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Initialize the XGBClassifier with label_encoder=False
    model = xgb.XGBClassifier(label_encoder=False, eval_metric='logloss')

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return round(accuracy*100,2)

print(XGBoost())

#d = pd.DataFrame(X,columns=data.columns.map(str))
#d['Classification'] = y

# Plot feature importance
#def Algos():
#    plot_importance(model, importance_type='gain', max_num_features=14)
#    plt.show()
#
## Plotting
#def Matrix():
#    cm = confusion_matrix(y_test, y_pred)
#    plt.figure(figsize=(10, 6))
#    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#    plt.xlabel('Predicted Label')
#    plt.ylabel('True Label')
#    plt.title('Confusion Matrix')
#    plt.show()

