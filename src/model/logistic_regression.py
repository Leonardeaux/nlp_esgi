import json
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression()

    """Logistic Regression Model"""
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def dump(self, filename_output):
        model_params = {
            "intercept": self.model.intercept_.tolist(),
            "coef": self.model.coef_.tolist(),
            "classes": self.model.classes_.tolist()
        }
        
        json_dump = json.dump(model_params, json_file)

        with open(filename_output, 'w') as json_file:
            json_dump

        return json_dump
