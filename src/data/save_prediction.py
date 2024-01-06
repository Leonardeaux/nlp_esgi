import pandas as pd
import ast


def save_prediction(model, X, df, output_filename, task):
    y_pred = model.predict(X)

    if task == "is_comic_video":
        df["prediction"] = y_pred.tolist()

    elif task == "is_name":
        pred = []
        for i, row in df.iterrows():
            targets = row['is_name']
            
            pred_line = []
            for j in range(len(targets)):
                pred_line.append(y_pred[i + j])

            pred.append(pred_line)

        df["prediction"] = pred
    elif task == "find_comic_name":
        df["prediction"] = y_pred
    

    df.to_csv(output_filename, index=False)
