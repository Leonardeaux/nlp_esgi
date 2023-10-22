import nltk
import string
import pandas as pd
import ast
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


def make_features(df, task, remove_ponct=True):
    nltk.download('stopwords')
    nltk.download('punkt')

    if task == "is_comic_video":
        X = df["video_name"]

        X = X.apply(preprocess_text(remove_ponct))

        y = df["is_comic"]

    elif task == "is_name":
        if remove_ponct:
            df['video_name'] = df['video_name'].apply(lambda x: remove_ponctuation(x))

        df['is_name'] = df['is_name'].apply(lambda x: ast.literal_eval(x))

        data = []

        for _, row in df.iterrows():
            words = row['video_name'].split()
            targets = row['is_name']

            for i, word in enumerate(words):
                line = [
                    word,                       # word
                    int(i == len(words) - 1),   # is final word
                    int(i == 0),                # is starting word
                    int(word[0].isupper())      # is capitalized
                ]
                try:
                    target = targets[i]
                except IndexError:
                    continue

                line.append(target)
                data.append(line)

        new_df = pd.DataFrame(data, columns=['word', 'is_final_word', 'is_starting_word', 'is_capitalized', 'target'])

        # X = new_df.drop(columns=['target']).to_dict(orient='records')
        X = new_df.drop(columns=['target']).drop(columns=['is_final_word']).drop(columns=['is_starting_word']).to_dict(orient='records')
        y = new_df["target"]

    elif task == "find_comic_name":
        y = df["comic_name"]
    else:
        raise ValueError("Unknown task")
    

    return X, y

# X, y = get_data(pd.read_csv("src/data/raw/train.csv"), "is_name")

# print(X.head(10))
# print(y.head(10))

# print(X.count())
# print(y.count())