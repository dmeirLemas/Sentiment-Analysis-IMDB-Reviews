import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def clean_text(text):
    text = text.lower()
    text = re.sub("<b />", "", text)
    text = re.sub(r"https\S+|www\S+|http\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@W+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    tokens = word_tokenize(text)

    text = [token for token in tokens if token not in set(stopwords.words("english"))]

    return " ".join(text)


stemmer = PorterStemmer()


def stemm(data):
    _ = [stemmer.stem(w) for w in data]
    return data


df = pd.read_csv("./IMDB Dataset.csv")

df["review"] = df["review"].apply(clean_text)

df["sentiment"].replace("positive", 1, inplace=True)
df["sentiment"].replace("negative", 0, inplace=True)


df["review"] = df["review"].apply(lambda x: stemm(x))

df = df.drop_duplicates("review")

df.to_csv("proc_IMDB.csv")
