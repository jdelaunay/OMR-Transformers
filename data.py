"""Module that create the datasets used as inputs to our models."""


import tensorflow as tf


# pylint: disable=W0621
CLASS_NAMES_NOTES = ["rest","A","A-","A#","B","B-","B#","C","C-","C#","D",
        "D-","D#","E","E-","E#","F","F-","F#","G","G-","G#"]


def decode_img(img_byte):
    """Decode an image from a binary representation to a tensor.

    Args:
        img_byte (tf.strings): the image to decode

    Returns:
         img (tf.dtype): the decoded image
    """
    img = tf.io.decode_png(img_byte, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def augment_dataset(img, label):
    """Apply data augmentation.

    Args:
        img (tf.Tensor): the template image
        label (float): label of  the image

    Returns:
        img (tf.Tensor): new image for data augmentation
        label (float): label of the image
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_hue(img, 0.08)
    img = tf.image.random_saturation(img, 0.6, 1.6)
    img = tf.image.random_brightness(img, 0.05)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    return img, label


def preprocess_img(img):
    """Apply preprocessing to the images. Empty for now.

    Args:
        img (tf.Tensor): image for preprocessing

    Returns:
        img (tf.Tensor): image after preprocessing
    """
    return img


def get_label(file_path):
    """Get label from file path given a class names.

    Args:
        file_path (str): the file path

    Returns:
        (tf.output_type): label encoded as a one-hot vector
    """
    label = tf.strings.split(file_path, "/")[-2]
    label = tf.cast(label == CLASS_NAMES, tf.int32)
    return tf.argmax(label, axis=0)


def process_path(file_path):
    """Define the data pipeline to process an image path.

    1. Get label from path
    2. Read file
    3. Decode from binary to image
    4. Apply preprocessing

    Args:
        file_path (str): the file path

    Returns:
        img (np.ndarray): image
        label (int): corresponding label
    """
    label = get_label(file_path)
    img_byte = tf.io.read_file(file_path)
    img = decode_img(img_byte)
    img = preprocess_img(img)
    return img, label


def ds_training_configuration(train_ds, params):
    """Describe the configuration of a training dataset.

    Args:
        train_ds (tf.data.Dataset): the train dataset
        params (dict): dictionnary of the parameters

    Returns:
       train_ds (tf.data.Dataset): the configured train dataset
    """
    train_ds = train_ds.shuffle(buffer_size=params["shuffle_buffer_size"])
    train_ds = train_ds.repeat(params["repeat"])
    if params["padding"]:
        train_ds = train_ds.padded_batch(
            params["batch_size"], padded_shapes=([None, None, 3], []),
        )
    else:
        train_ds = train_ds.batch(batch_size=params["batch_size"])
    train_ds = train_ds.prefetch(buffer_size=params["prefetch_buffer_size"])
    return train_ds


def ds_val_test_configuration(val_ds, params):
    """Describe the configuration of a validation/test dataset.

    Args:
        ds (tf.data.Dataset): the dataset
        params (dict): a dictionary with all the parameters

    Returns:
        ds (tf.data.Dataset): configured validation_test dataset
    """
    if params["padding"]:
        val_ds = val_ds.padded_batch(
            params["batch_size"], padded_shapes=([None, None, 3], []),
        )
    else:
        val_ds = val_ds.batch(batch_size=params["batch_size"])
    return val_ds


def create_training_ds(params):
    """Create a training dataset given class names and a dict of params.

    Args:
        params (dict): a dictionary with all the parameters

    Returns:
        train_ds (tf.data.Dataset): the training dataset
    """
    if params["train_data_path"][-1] != "/":
        params["train_data_path"] += "/"
    train_ds = tf.data.Dataset.list_files(params["train_data_path"] + "*/*")
    train_ds = train_ds.map(process_path)
    if params["augment"]:
        train_ds = train_ds.map(augment_dataset)
    train_ds = ds_training_configuration(train_ds, params)
    return train_ds


def create_val_ds(params):
    """Create a validation dataset given class names and a dict of params.

    Args:
        params (dict): a dictionary with all the parameters

    Returns:
        val_ds (tf.data.Dataset): the validation dataset
    """
    if params["val_data_path"][-1] != "/":
        params["val_data_path"] += "/"
    val_ds = tf.data.Dataset.list_files(params["val_data_path"] + "*/*")
    val_ds = val_ds.map(process_path)
    val_ds = ds_val_test_configuration(val_ds, params)
    return val_ds


def create_test_ds(params):
    """Create a test dataset given class names and a dict of params.

    Args:
        params (dict): a dictionary with all the parameters

    Returns:
        test_ds(tf.data.Dataset): the test dataset
    """
    if params["test_data_path"][-1] != "/":
        params["test_data_path"] += "/"
    test_ds = tf.data.Dataset.list_files(params["test_data_path"] + "*/*")
    test_ds = test_ds.map(process_path)
    test_ds = ds_val_test_configuration(test_ds, params)
    return test_ds


if __name__ == "__main__":
    import cv2

    params = {
        "train_data_path": "/data/casia/CASIA/train/",
        "val_data_path": "/data/casia/CASIA/val/",
        "test_data_path": "/data/casia/CASIA/test/",
        "batch_size": 1,
        "shuffle_buffer_size": 200,
        "prefetch_buffer_size": 1,
        "repeat": 1,
        "padding": True,
        "augment": True
    }
    ds = create_training_ds(params)
    i = 0
    for img, label in ds.take(10):
        print(label)
        print(img.numpy()[0])
        cv2.imwrite(
            "/data/test{}_{}.png".format(i, label.numpy()),
            cv2.cvtColor(img.numpy()[0] * 255, cv2.COLOR_BGR2RGB)
        )
        i += 1

