import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from xgboost import plot_importance
import seaborn as sns

# Load dataset
data = pd.read_csv("Data.csv")
X = data.drop(columns=['Text'])
y = data['Classification'] = data['Classification'] - 1

N = 20
accavg = 0

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the XGBClassifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
accavg += accuracy

print(f"Average Accuracy: {accavg*100/N:.2f}%")

cm = confusion_matrix(y_test, y_pred)

# Plotting
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Plot feature importance
plot_importance(model, importance_type='gain', max_num_features=6)
plt.show()

# Print feature names
for i, f in enumerate(X.columns):
    print(f'f{i:02d} ---> {f}')