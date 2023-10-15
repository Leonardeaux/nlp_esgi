import click
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score

from data.make_dataset import make_dataset
from features.make_features import make_features
from model.main import make_model

@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="src/data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="src/model/random_forest.gzip", help="File to dump model")
@click.option("--model_name", default="random_forest", help="Name of the model to use")
def train(task, input_filename, model_dump_filename, model_name):
    df = make_dataset(input_filename)
    X, y = make_features(df, task)

    model = make_model(model_name)
    model.fit(X, y)

    return joblib.dump(model, model_dump_filename)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="src/data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="src/model/random_forest.gzip", help="File to dump model")
@click.option("--output_filename", default="src/data/processed/random_forest_prediction.csv", help="Output file for predictions")
def test(task, input_filename, model_dump_filename, output_filename):
    df = make_dataset(input_filename)

    X, y = make_features(df, task)

    model = joblib.load(model_dump_filename)

    y_pred = model.predict(X)

    df["prediction"] = y_pred

    df.to_csv(output_filename, index=False)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="src/data/raw/train.csv", help="File training data")
@click.option("--model_name", default="random_forest", help="Name of the model to use")
def evaluate(task, input_filename, model_name):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df, task)

    # Object with .fit, .predict methods
    model = make_model(model_name)

    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    # Scikit learn has function for cross validation
    scores = cross_val_score(model, X, y, scoring="accuracy")

    print(f"Got accuracy {100 * np.mean(scores)}%")

    return scores


cli.add_command(train)
cli.add_command(test)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
