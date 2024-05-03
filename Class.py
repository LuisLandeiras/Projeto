import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv("Data.csv")
A = data.drop(columns=['Text'])

Algos = ['ARI','Coleman','Grade','Grammar','LexicalDensity','LexicalDiversity','Reading','SentenceLength','SentimentNeg','SentimentNeu','SentimentPos','Smog','Tree','WordLength']

for Algo in Algos:
    print("----------------------------"+Algo+"--------------------------")
    b = data[Algo]
    A_train, A_test, b_train, b_test = train_test_split(A,b,test_size=0.1)

    model = DecisionTreeRegressor()
    model.fit(A_train, b_train)

    print("Test values:")
    print(A_test)

    print("These are predictions for inputs:")
    predictions = model.predict(A_test)
    print(predictions)

    mse = mean_squared_error(b_test, predictions)
    print("Mean Squared Error:")
    print(mse)

