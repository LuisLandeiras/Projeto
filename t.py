import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer  # Example text transformer
import numpy as np

def XGBoost(data_path="Data.csv"):
    # Load the data
    data = pd.read_csv(data_path)

    # Define feature set and target variable
    X = data.drop(columns=['Text', 'Classification', 'SentimentNeg', 'SentimentNeu', 'SentimentPos'])
    y = data['Classification']

    # Example transformation of text to features
    text_data = data['Text']
    vectorizer = TfidfVectorizer()
    text_features = vectorizer.fit_transform(text_data)

    # Combine text features with other features
    X_combined = np.hstack((X.values, text_features.toarray()))

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.1, random_state=42)

    # Initialize the XGBClassifier with label_encoder=False
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Return model, vectorizer, and accuracy for further use
    return model, vectorizer, round(accuracy, 2)

def predict_text(model, vectorizer, text, other_features):
    # Transform the text input
    text_feature = vectorizer.transform([text])

    # Combine with other features
    combined_features = np.hstack((other_features, text_feature.toarray()))

    # Make a prediction
    prediction = model.predict(combined_features)

    return prediction[0]

# Example usage
model, vectorizer, accuracy = XGBoost("Data.csv")
print("Model Accuracy:", accuracy)

# Predicting a new text
text_to_predict = "Your text input here"
# Assuming other features are available as an array, e.g., from user input or defaults
other_features = np.array([value1, value2, value3, ...])  # Fill with actual values

# Call predict_text to get the prediction
prediction = predict_text(model, vectorizer, text_to_predict, other_features)
print("Predicted Classification:", prediction)
