import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv("Data.csv")

Algos = ['ARI','Coleman','Grade','Grammar','LexicalDensity','LexicalDiversity','Reading','SentenceLength','SentimentNeg','SentimentNeu','SentimentPos','Smog','Tree','WordLength','Classification']

A_train, A_test, b_train, b_test = train_test_split(data.drop(columns=['Text']), data[Algos], test_size=0.05)

for Algo in Algos:
    print("----------------------------"+Algo+"----------------------------")
    b = b_test[Algo]

    model = DecisionTreeRegressor()
    model.fit(A_train, b_train[Algo])

    print("Test values:")
    print(A_test)

    print("These are predictions for inputs:")
    predictions = model.predict(A_test)
    print(predictions)

    mse = mean_squared_error(b, predictions)
    print("Mean Squared Error:")
    print(mse)
