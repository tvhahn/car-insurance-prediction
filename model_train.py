import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorboard
import os
from pathlib import Path
import datetime
from sklearn.model_selection import train_test_split

data_path = Path("data")

# make a list of all the csv's in the data folder
file_csv = []
for file in os.listdir(data_path):
    if file.endswith(".csv"):
        file_csv.append(file)


def prepare_csv(csv_path):
    df = pd.read_csv(csv_path)

    # scale some of the large values
    df["age"] = df["age"] / 100.0
    df["dist"] = df["dist"] / 25000.0
    df["year"] = df["year"] / 2020.0
    df = df.drop(["model"], axis=1, inplace=False)

    # y-value
    y = df["quote"].to_numpy()
    print(y.shape)
    df = df.drop(["quote"], axis=1, inplace=False)
    df = pd.get_dummies(df, drop_first=True)
    col_list = list(df.columns)
    X = df.to_numpy()

    # reshape the X to the proper shape
    X = np.reshape(X, (X.shape[0], -1, X.shape[1]))
    return X, y


# train the models
# split up the X and y for both the quote prediction and confidence interval and
for f in file_csv:

    epoch_no = 500
    earl_stop_no = 30

    company_name = f.split(".")[0]

    print("PREPARING DATA FOR ", company_name)
    X, y = prepare_csv(data_path / f)

    X_quote = X[0 : int(X.shape[0] / 2)]
    y_quote = y[0 : int(y.shape[0] / 2)]
    print(X_quote.shape)
    print(y_quote.shape)
    X_conf = X[int(X.shape[0] / 2) :]
    y_conf = y[int(y.shape[0] / 2) :]
    print(X_conf.shape)
    print(y_conf.shape)

    X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(
        X_quote, y_quote, test_size=0.33, random_state=42
    )
    X_train_q, X_val_q, y_train_q, y_val_q = train_test_split(
        X_train_q, y_train_q, test_size=0.33, random_state=42
    )

    print("X_train_q shape:", X_train_q.shape, "\ty_train_q shape:", y_train_q.shape)
    print("X_val_q:", X_val_q.shape, "\t\t\ty_val_q shape:", y_val_q.shape)
    print("X_test_q:", X_test_q.shape, "\t\ty_test_q shape:", y_test_q.shape)

    #!#!#!#!#!#!#!#!#!#!#!#!#!#

    # build the model for predicting the quote
    model = keras.models.Sequential(
        [
            keras.layers.Dense(200, activation="relu", input_shape=[1, 114]),
            keras.layers.Dense(200, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(50, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation="linear"),
        ]
    )

    model.compile(loss="mse", optimizer="adam", metrics=["mse"])

    # create a name for the model so that we can track it in tensorboard
    log_dir = (
        "logs/"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        + "_quote_{}".format(company_name)
    )

    # create tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=0, update_freq="epoch", profile_batch=0
    )

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="mse", patience=earl_stop_no, restore_best_weights=True, verbose=1,
    )

    # fit
    history = model.fit(
        X_train_q,
        y_train_q,
        epochs=epoch_no,
        verbose=1,
        validation_data=(X_val_q, y_val_q),
        callbacks=[tensorboard_callback, earlystop_callback],
    )
    # save model
    model_save_name = "{}_model_quote.h5".format(company_name)
    model.save(model_save_name)

    #!#!#!#!#!#!#!#!#!#!#!#!#!#

    # now get the y-confidence predictions
    y_predict_conf = model.predict(X_conf)
    y_predict_conf = np.reshape(y_predict_conf, (y_predict_conf.shape[0],))
    y_conf_error = np.divide(np.abs(y_conf - y_predict_conf), y_conf)

    # prepare the data to train the confidence prediction
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_conf, y_conf_error, test_size=0.33, random_state=42
    )
    X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(
        X_train_c, y_train_c, test_size=0.33, random_state=42
    )

    # create a name for the model so that we can track it in tensorboard
    log_dir = (
        "logs/"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        + "_quote_{}".format(company_name)
    )

    # create tensorboard callback
    tensorboard_callback_2 = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=0, update_freq="epoch", profile_batch=0
    )

    earlystop_callback_2 = tf.keras.callbacks.EarlyStopping(
        monitor="mse", patience=earl_stop_no, restore_best_weights=True, verbose=1,
    )

    model_conf = keras.models.Sequential(
        [
            keras.layers.Dense(200, activation="relu", input_shape=[1, 114]),
            keras.layers.Dense(200, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(50, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model_conf.compile(loss="mse", optimizer="adam", metrics=["mse"])

    # create a name for the model so that we can track it in tensorboard
    log_dir = (
        "logs/"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        + "_conf_{}".format(company_name)
    )

    history = model_conf.fit(
        X_train_c,
        y_train_c,
        epochs=epoch_no,
        verbose=1,
        validation_data=(X_val_c, y_val_c),
        callbacks=[tensorboard_callback_2, earlystop_callback_2],
    )

    # save model
    model_save_name = "{}_model_conf.h5".format(company_name)
    model_conf.save(model_save_name)
