import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

df = pd.read_csv("proc_IMDB.csv")


X = df["review"]
y = df["sentiment"]

vector = TfidfVectorizer()
X = vector.fit_transform(X)

train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearSVC()
model.fit(train_X, train_y)

y_pred = model.predict(test_X)

accuracy = accuracy_score(test_y, y_pred)

print("Accuracy: ", accuracy)

out = pd.DataFrame({"sentiment": test_y, "prediction": y_pred})
out.to_csv("out.csv")
