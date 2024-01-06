import nltk
import string
import pandas as pd
import ast
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer

def remove_ponctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def preprocess_text(text, remove_ponct=True):
    stemmer = FrenchStemmer()
    stop_words = set(stopwords.words('french'))
    
    words = word_tokenize(text)
    filtered_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]

    final_text = " ".join(filtered_words)

    if remove_ponct:
        final_text = remove_ponctuation(final_text)

    return final_text

def make_feature_is_comic_video(df, remove_ponct):
    X = df["video_name"]

    X = X.apply(lambda x: preprocess_text(x, remove_ponct))

    y = df["is_comic"]

    return X, y

def make_feature_is_name(df, remove_ponct):
    if remove_ponct:
        df['video_name'] = df['video_name'].apply(lambda x: remove_ponctuation(x))

    df['is_name'] = df['is_name'].apply(lambda x: ast.literal_eval(x))

    data = []

    for row_nb, row in df.iterrows():
        words = re.split(r"[ ']", row['video_name'])
        targets = row['is_name']

        for i, word in enumerate(words):
            line = [
                word,                       # word
                int(i == len(words) - 1),   # is final word
                int(i == 0),                # is starting word
                int(words[0].isupper())     # is capitalized
            ]
            try:
                target = targets[i]
            except IndexError:
                print(f'Error in dataset, number of word and number of target does not match in row {row_nb} et la phrase : \n {row["video_name"]}')

            line.append(target)
            data.append(line)

    new_df = pd.DataFrame(data, columns=['word', 'is_final_word', 'is_starting_word', 'is_capitalized', 'target'])

    X = new_df.drop(columns=['target']).to_dict(orient='records')

    y = new_df["target"]

    y = y.rename('is_name')
    return X, y

def make_features(df, task, remove_ponct=True):

    nltk.download('stopwords')
    nltk.download('punkt')

    if task == "is_comic_video":
        X, y = make_feature_is_comic_video(df, remove_ponct)

    elif task == "is_name":
        X, y = make_feature_is_name(df, remove_ponct)

    elif task == "find_comic_name":
        X_is_name, y_is_name = make_feature_is_name(df, remove_ponct)

        X_is_comic, y_is_comic = make_feature_is_comic_video(df, remove_ponct)

        X = (X_is_name, X_is_comic, df["video_name"])
        y = (y_is_name, y_is_comic)

    else:
        raise ValueError("Unknown task")
    

    return X, y