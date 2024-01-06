import click
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score
from data.make_dataset import make_dataset
from data.save_prediction import save_prediction
from features.make_features import make_features
from model.main import make_model
from utils import get_index


@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="src/data/raw/train.csv", help="File training data")
@click.option("--model_name", default="random_forest", help="Name of the model to use")
def train(task, input_filename, model_name):
    df = make_dataset(input_filename)
    X, y = make_features(df, task)

    model = make_model(model_name, task)
    model.fit(X, y)

    return joblib.dump(model, f"src/model/{model_name}.gzip")


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="src/data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="src/model/random_forest.gzip", help="File to dump model")
@click.option("--output_filename", default="src/data/processed/random_forest_prediction.csv", help="Output file for predictions")
def test(task, input_filename, model_dump_filename, output_filename):
    df = make_dataset(input_filename)

    X, _ = make_features(df, task)

    model = joblib.load(model_dump_filename)

    save_prediction(model, X, df, output_filename, task)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="src/data/raw/train.csv", help="File training data")
@click.option("--model_name", default="random_forest", help="Name of the model to use")
def evaluate(task, input_filename, model_name):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df, task, remove_ponct=True)

    if task != "find_comic_name":
        # Object with .fit, .predict methods
        model = make_model(model_name, task)

        # Run k-fold cross validation. Print results
        return evaluate_model(model, X, y)
    else:
        # Object with .fit, .predict methods
        model = make_model(model_name, task)

        # Run k-fold cross validation. Print results
        return evaluate_model_for_task_3(model, X, y, df)


def evaluate_model(model, X, y):
    # Scikit learn has function for cross validation
    scores = cross_val_score(model, X, y, scoring="accuracy")

    print(f"Got accuracy {100 * np.mean(scores)}%")

    return scores


def evaluate_model_for_task_3(model, X, y, df):

    model.fit(X, y)

    y_pred = model.predict(X)

    if len(y_pred) != len(df['comic_name']):
        raise ValueError("Les listes de prédictions et de vrais labels doivent avoir la même longueur.")
    
    correct_predictions = 0
    total_predictions = len(y_pred)

    for i, row in df.iterrows():
        true_target = get_index(row['video_name'], row['comic_name'])

        try:
            y_pred[i] = y_pred[i].tolist()
        except:
            pass

        if y_pred[i] == true_target:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions

    print(f"Got accuracy {accuracy}%")

    return accuracy


cli.add_command(train)
cli.add_command(test)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
