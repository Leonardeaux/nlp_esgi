import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer

def preprocess_text(text):
    stemmer = FrenchStemmer()
    stop_words = set(stopwords.words('french'))
    
    words = word_tokenize(text)
    filtered_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


def make_features(df, task):
    nltk.download('stopwords')

    y = get_output(df, task)

    X = df["video_name"]

    print(X.head())

    X = X.apply(preprocess_text)

    print(X.head())

    return X, y


def get_output(df, task):
    if task == "is_comic_video":
        y = df["is_comic"]
    elif task == "is_name":
        y = df["is_name"]
    elif task == "find_comic_name":
        y = df["comic_name"]
    else:
        raise ValueError("Unknown task")

    return y
