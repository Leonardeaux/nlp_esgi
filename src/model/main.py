from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

def make_model(model_name: str):
    pipeline_step = []

    pipeline_step.append(("count_vectorizer", CountVectorizer()))

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
