#!/usr/bin/env python
# coding: utf-8

import argparse
from datetime import date

import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline


def read_dataframe(filename: str) -> pd.DataFrame:
    """Reads the dataframe from a file/url and does some feature engineering

    Args:
        filename (str): from where to get the data

    Returns:
        pd.DataFrame: Processed df
    """
    df = pd.read_parquet(filename)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df


def train(train_date: date, val_date: date, out_path: str) -> None:
    """
    Train a linear regression model on trip data and save the trained pipeline.

    Loads training and validation datasets, preprocesses features, trains
    a model to predict trip duration, evaluates it on the validation set,
    and serializes the trained pipeline to disk.

    Args:
        train_date: Month/year of the training dataset.
        val_date: Month/year of the validation dataset.
        out_path: Path where the trained model will be saved.
    """
    
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
    train_url = base_url.format(year=train_date.year, month=train_date.month)
    val_url = base_url.format(year=val_date.year, month=val_date.month)

    df_train = read_dataframe(train_url)
    df_val = read_dataframe(val_url)

    print(f"train size: {len(df_train)}, val size: {len(df_val)}")

    categorical = ["PULocationID", "DOLocationID"]
    numerical = ["trip_distance"]

    dv = DictVectorizer()
    lr = LinearRegression()
    pipeline = make_pipeline(dv, lr)

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    val_dicts = df_val[categorical + numerical].to_dict(orient="records")

    target = "duration"
    y_train = df_train[target].values
    y_val = df_val[target].values

    pipeline.fit(train_dicts, y_train)
    y_pred = pipeline.predict(val_dicts)

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"MSE: {mse}")

    with open(out_path, "wb") as f_out:
        pickle.dump(pipeline, f_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model based on specified dates and save it to a given path")
    parser.add_argument("--train-date", required=True, help="Train month in the YYYY-MM format")
    parser.add_argument("--val-date", required=True, help="Validation month in the YYYY-MM format")
    parser.add_argument("--model-save-path", required=True, help="Path where the trained model is saved.")
    
    args = parser.parse_args()
    train_year, train_month = args.train_date.split("-")
    val_year, val_month = args.val_date.split("-")

    train_date = date(int(train_year), int(train_month), 1)
    val_date = date(int(val_year), int(val_month), 1)
    out_path = args.model_save_path
    train(train_date=train_date, val_date=val_date, out_path=out_path)
