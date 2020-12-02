"""Describes functions to train a neural network model"""

import importlib
from os.path import isdir, join, exists
from os import makedirs

import tensorflow as tf
import tensorflow.keras.callbacks as cb

from data import DataGenerator

MODEL_FUNC = "get_model"


def train(params):
    """Train a model using a set of parameters

    Args:
        params(dict): parameters to train the model

    Returns:
        (list): results of the metrics
        history(list): loss and metrics
        model(tf.keras.Model): the trained model
    """

    # Verify paths
    assert isdir(params["train_data_path"]), "Training dir not found"
    assert isdir(params["val_data_path"]), "Validation dir not found"
    assert isdir(params["test_data_path"]), "Test dir not found"

    # Create directory for model path
    if not exists(params["model_path"]):
        makedirs(params["model_path"])
    model_path = join(params["model_path"], params["name"]) + ".h5"

    # Load modules
    networks_module = importlib.import_module(f"networks.{params['model']}")
    metrics_module = importlib.import_module("metrics")

    # Create the train, val and test datasets
    train_generator = DataGenerator(params["train_data_path"],
                                          "train",
                                          batch_size = params["batch_size"],
                                          aug_rate = 0.25
                                         )
    valid_generator = DataGenerator(params["val_data_path"],
                                          "validation",
                                          batch_size = params["batch_size"],
                                          aug_rate = 0.25
                                         )

    # Load the network
    model = getattr(networks_module, MODEL_FUNC)(params)
    model.compile(
        loss={'output_notes': 'categorical_crossentropy',
              'output_octaves': 'categorical_crossentropy',
              'output_rythms': 'categorical_crossentropy'},
        optimizer=tf.keras.optimizers.Adam(lr=params["learning_rate"]),
        metrics=getattr(metrics_module, "get_metrics")()
    )

    # Callbacks
    callbacks = [
        cb.Tensorboard(
            join(params["log_path"], params["name"])
        ),
        cb.EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            restore_best_weights=True,
        ),
        cb.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        cb.ReduceLROnPlateau(
            monitor="val_loss",
            factor=params["reduce_lr_factor"],
            patience=params["reduce_lr_patience"],
            verbose=1,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=0
        ),
    ]

    # Train and evaluate the model
    history = model.fit(
        train_ds,
        epochs=params["epochs"],
        validation_data=val_ds,
        callbacks=callbacks,
    )

    return model.evaluate(test_ds), history, model
