# Importing the usual libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
 
data = pd.read_csv("Data.csv")
A = data.drop(columns=['Text'])
b = data['ARI']

A_train, A_test, b_train, b_test = train_test_split(A,b,test_size=0.1)

model = DecisionTreeRegressor()
model.fit(A_train, b_train)

print("Test values:")
print(A_test)

print("These are predictions for inputs:")
predictions = model.predict(A_test)
print(predictions)

#score = accuracy_score(b_test, predictions)
#print("Accuracy:")
#print(score)

mse = mean_squared_error(b_test, predictions)
print("Mean Squared Error:")
print(mse)

