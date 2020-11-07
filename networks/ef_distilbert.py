"""This module contains the class for multi-tasks learning"""

from keras_ctcmodel.CTCModel import CTCModel as CTCModel
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers import (
    Inputs,
    Conv2D,
    Dense,
    Flatten,
    TimeDistributed
)
import transformers


def build_head(inputs, backbone, n_classes, name):
    """
    Build a head for only one task.

    Args:
        input_data(): the Inputs
        backbone(tf.keras.Model): the backbone of the model
        name(str): the name of the task
    Returns:
        (tf.keras.Model): the Model
    """
    config = transformers.DistilBertConfig(vocab_size=n_classes)
    distil_bert = transformers.TFDistilBertModel(config)(backbone)
    dense = TimeDistributed(
        Dense(n_classes, activation="softmax")
    )(distil_bert)
    return CTCModel([inputs], [dense], name=name)


def get_model(params):
    """
    Create a model from parameters.
    Args:
        params(dict): the parameters of the model
    Returns:
        (tf.keras.Model): the model
    """
    inputs = Inputs(shape=(1000, 159, 2))
    backbone = efficientnet.EfficientNetB2(
        include_top=False
    )(inputs)
    conv = Conv2D(256, (1, 1), padding="same", activation="relu")(backbone)
    flatten = Flatten()(conv)

    notes_model = build_head(inputs, flatten, 23, "notes")
    octaves_model = build_head(inputs, flatten, 15, "octaves")
    rythms_model = build_head(inputs, flatten, 60, "rythms")

    return notes_model, octaves_model, rythms_model
