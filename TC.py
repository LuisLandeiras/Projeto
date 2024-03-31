# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import AuxFun

Texto = AuxFun.File("Textos/obama.txt")
Texto1 = AuxFun.File("Textos/trump.txt")
Texto2 = AuxFun.File("Textos/biden.txt")
Texto3 = AuxFun.File("Textos/The_Mother.txt")
Texto4 = AuxFun.File("Textos/Men_Without_Women.txt")

Lista = []
Lista.append(Texto)
Lista.append(Texto1)
Lista.append(Texto2)
Lista.append(Texto3)
#Lista.append(Texto4)

# Labels for the sample data
labels = ["Nation", "Family", "Country", "Mother"]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Lista, labels, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Initialize and train the classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectors, y_train)

# Predictions
predictions = classifier.predict(X_test_vectors)

# Evaluate the classifier
print("Accuracy:", accuracy_score(y_test, predictions))
#print("\nClassification Report:")
print(classification_report(y_test, predictions))
