import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance

data = pd.read_csv("Data.csv")

X = data.drop(columns=['Text','Classification'])
y = data['Classification']

N = 20
accavg = 0

for _ in range(N):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Initialize the XGBClassifier with label_encoder=False
    model = DecisionTreeRegressor()

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accavg += accuracy
    
def plot_feature_importance(model, feature_names, max_num_features=14):
    feature_importance = model.feature_importances_
    sorted_idx = feature_importance.argsort()[-max_num_features:]
    
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.show()

print(accavg)
#plot_feature_importance(model, feature_names=range(X.shape[1]))
