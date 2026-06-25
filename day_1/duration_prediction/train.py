#!/usr/bin/env python
# coding: utf-8

from datetime import date

import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from loguru import logger

def read_dataframe(filename: str) -> pd.DataFrame:
    """Reads the dataframe from a file/url and does some feature engineering

    Args:
        filename (str): from where to get the data

    Returns:
        pd.DataFrame: Processed df
    """

    try:
        df = pd.read_parquet(filename)

        df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)]

        categorical = ["PULocationID", "DOLocationID"]
        df[categorical] = df[categorical].astype(str)

        return df
    except Exception as e:
        logger.exception(f"Error loading the data for {filename}. error: {e}")
        raise e


def train(train_date: date, val_date: date, out_path: str) -> float:
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

    logger.debug(f"train size: {len(df_train)}, val size: {len(df_val)}")

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
    logger.info(f"MSE: {mse}")

    with open(out_path, "wb") as f_out:
        pickle.dump(pipeline, f_out)
        
    return mse

