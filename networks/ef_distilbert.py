"""This module contains the model for multi-task learning."""

from keras_ctcmodel.CTCModel import CTCModel as CTCModel
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.layers import (
    Inputs,
    Conv2D,
    Dense,
    MaxPool2D,
    TimeDistributed,
    Bidirectional,
    LSTM,
    Reshape
)


def get_model(params):
    """Create a multitasks model from parameters.

    Args:
        params(dict): the parameters of the model
    Returns:
        (tf.keras.Model): the model
    """

    inputs = Inputs(shape=(1000, 159, 3))
    backbone = mobilenet_v2.MobileNetV2(
        include_top=False
    )(inputs)
    conv = Conv2D(256, (1, 1), padding="same", activation="relu")(backbone)
    pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv)
    reshape  = Reshape((30, 256))(pool)

    blstm_1 = Bidirectional(LSTM(params["lstm_size"],
                                 return_sequences=True,
                                 dropout=params["dropout"]
                                 )
                           )(reshape)
    blstm_2 = Bidirectional(LSTM(params["lstm_size"],
                                 return_sequences=True,
                                 dropout=params["dropout"]
                                )
                           )(blstm_1)

    notes_dense = TimeDistributed(
        Dense(24, activation="softmax")
    )(blstm_2)
    octaves_dense = TimeDistributed(
        Dense(16, activation="softmax")
    )(blstm_2)
    rythms_dense = TimeDistributed(
        Dense(61, activation="softmax")
    )(blstm_2)

    return CTCModel([inputs], [notes_dense, octaves_dense, rythms_dense], name=params["name"])
