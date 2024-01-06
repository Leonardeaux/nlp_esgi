import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

class CombinedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_is_comic = make_model(f'{model_name}', "is_comic_video")
        self.model_is_name = make_model(f'{model_name}', "is_name")

    def fit(self, X, y):
        X_is_comic = X[1]
        y_is_comic = y[1]
        self.model_is_comic.fit(X_is_comic, y_is_comic)

        X_is_name = X[0]
        y_is_name = y[0]
        self.model_is_name.fit(X_is_name, y_is_name)

    def predict(self, X):
        predictions_is_comic = self.model_is_comic.predict(X[1])
        predictions_is_name = self.model_is_name.predict(X[0])

        cpt = 0
        is_name_pred_array = []
        for i in range(len(X[2])):
            words_nb = len(re.split(r"[ ']", X[2][i]))
            if predictions_is_comic[i] == 1:

                is_name_pred_array.append(predictions_is_name[cpt:cpt + words_nb])

            else:
                is_name_pred_array.append([0] * words_nb)

            cpt = cpt + words_nb

        return is_name_pred_array


def make_model(model_name: str, task: str):
    pipeline_step = []

    if task == "is_comic_video":
        pipeline_step.append(("count_vectorizer", CountVectorizer()))
    elif task == "is_name":
        pipeline_step.append(("dict_vectorizer", DictVectorizer(sparse=False)))
    elif task == "find_comic_name":
        return CombinedModel(model_name)

    if model_name == "random_forest":
        pipeline_step.append(("random_forest", RandomForestClassifier(random_state=42)))
    elif model_name == "logistic_regression":
        pipeline_step.append(("logistic_regression", LogisticRegression(random_state=42)))
    elif model_name == "svm":
        pipeline_step.append(("svm", SVC(random_state=42)))
    elif model_name == "naive_bayes":
        pipeline_step.append(("naive_bayes", MultinomialNB()))
    elif model_name == "xgboost":
        pipeline_step.append(("xgboost", GradientBoostingClassifier(random_state=42)))
    else:
        raise ValueError("Unknown model")
    
    return Pipeline(pipeline_step)