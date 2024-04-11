import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.svm import LinearSVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import gc

df = pd.read_csv("proc_IMDB.csv", index_col="id")


X = df["review"]
y = df["sentiment"]

vector = TfidfVectorizer()
X = vector.fit_transform(X)

train_X, test_X, train_y, test_y, new_train_y, test_y_ = train_test_split(
    X, y, df["ratings"], test_size=0.2, random_state=42
)

model = LinearSVC()
model.fit(train_X, train_y)

y_pred = model.predict(test_X)

accuracy = accuracy_score(test_y, y_pred)
print("Accuracy: ", accuracy)

del model
del df
del vector
gc.collect()

scaler = MinMaxScaler(feature_range=(-5, 5))
scaler.fit(new_train_y.to_frame().values.reshape(-1, 1))

new_train_y = scaler.transform(new_train_y.to_frame().values.reshape(-1, 1))
test_y_ = scaler.transform(test_y_.to_frame().values.reshape(-1, 1))

svr = SVR(max_iter=1000)
svr.fit(train_X, new_train_y)

new_preds = svr.predict(test_X)

mse = mean_squared_error(test_y_, new_preds)
print(mse)


out = pd.DataFrame(
    {
        "sentiment": test_y,
        "sent_pred": y_pred,
        "ratings": test_y_.flatten(),
        "rating_pred": new_preds,
    }
)
out.loc[(out["sentiment"] == 1) & (out["ratings"] < 0), "ratings"] *= -1
out.loc[(out["sentiment"] == 1) & (out["rating_pred"] < 0), "rating_pred"] *= -1

out.to_csv("out.csv", index_label="id")
