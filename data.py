"""Module that create the datasets used as inputs to our models."""


import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences


# pylint: disable=W0621
CLASS_NAMES_NOTES = ["rest","A","A-","A#","B","B-","B#","C","C-","C#","D",
        "D-","D#","E","E-","E#","F","F-","F#","G","G-","G#"]
CLASS_NAMES_OCTAVES = ['6', '2', '1', '9', '7', '0', '|', 'rest', '5', '11',
        '-1', '10', '4', '8', '3']
CLASS_NAMES_RYTHMS = ['0.75', '10.0', '0.0', '0.375', '1', '2/3', '7/6',
        '0.25', '0', '0.125', '2.0', '0.5', '0.09375', '1.0', '2.5', 'EOF',
        '8.0', '7/20', '4.0', '3.5', '1/9', '4/3', '7.5', '7.0', '12.0',
        '1/5', '0.875', '4.5', '9.0', '1/10', '1.875', '8/3', '0.03125',
        '5.5', '16.0', '3.0', '1/20', '4/5', '5.0', '0.0234375', '2/5',
        '0.1875', '1.75', '1/7', '6.0', '4/7', '1/3', '0.4375', '1/12', '-1',
        '0.046875', '7/10', '2.25', '1.5', '0.0625', '1/24', '24.0', '|',
        '1/6', '2', '28.0']


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 list_IDs,
                 directory,
                 batch_size=params["batch_size"],
                 dim=(params["batch_size"],
                      params["batch_size"],
                      params["batch_size"]),
                 n_channels=1,
                 shuffle=True, aug_rate = 0, out="all"):
        'Initialization'
        self.directory = directory
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.label_file = open("../data/{}_labels.txt".format(directory),'r')
        self.list_label_notes = CLASS_NAMES_NOTES
        self.list_label_octaves = CLASS_NAMES_OCTAVES
        self.list_label_rythms = CLASS_NAMES_RYTHMS
        self.aug_rate = aug_rate
        self.category_function = self.notes_label
        self.n_classes = 0


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if (index + 1) * self.batch_size < len(self.indexes): 
            indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        else:
            indexes = self.indexes[index * self.batch_size : -1]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        list_label = dict()
        list_label["notes_labels"] = self.notes_label
        list_label["octaves_labels"] = self.octaves_label
        list_label["rythms_labels"] = self.rythms_label

        # Generate data
        X = self.__data_generation(list_IDs_temp, list_label)
        y = np.zeros(len(X[2]))
        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp, list_label):
        'Generates data containing batch_size samples'
        # Initialization
        images_path = os.path.abspath("../data/{}_out_x/".format(self.directory))
        X = list()
        y = dict()
        y["notes_labels"] = list()
        y["octaves_labels"] = list()
        y["rythms_labels"] = list()
        X_len = list()
        y_len = list()
        fail = 0
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            i = i-fail
            y_notes_temp = np.asarray(list(self.notes_label(self.label_file , ID)), dtype="str")
            y_octaves_temp = np.asarray(list(self.octaves_label(self.label_file , ID)), dtype="str")
            y_rythms_temp = np.asarray(list(self.rythms_label(self.label_file , ID)), dtype="str")

            image_temp = cv2.imread(images_path +'/'+ ID, cv2.IMREAD_UNCHANGED )
            if len(image_temp.shape) > 2:
                    image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2GRAY) 
            image_temp = cv2.resize(image_temp,(0,0),fx=0.5,fy=0.5, interpolation =cv2.INTER_CUBIC)
            image_temp = cv2.resize(image_temp,(0,0),fx=0.5,fy=0.5, interpolation =cv2.INTER_CUBIC).T
            # self.augment_image(image_temp)
            X_len.append(image_temp.shape[0])

            if y_notes_temp.shape[0] != 0:    
                y["notes_labels"].append(self.convert_into_number(y_notes_temp, list_label))
                y_notes_len.append(y_temp.shape[0])
            if y_octaves_temp.shape[0] != 0:    
                y["octaves_labels"].append(self.convert_into_number(y_octaves_temp, list_label))
                y_notes_len.append(y_octaves_temp.shape[0])
            if y_rythms_temp.shape[0] != 0:    
                y["rythms_labels"].append(self.convert_into_number(y_rythms_temp, list_label))
                y_rythms_len.append(y_rythms_temp.shape[0])

            # Store sample
            X.append(image_temp)
                
        y_len = np.asarray(y_len)
        X_len = np.asarray(X_len)
       
        pad_value = max(X_len)
        for i in range(len(X)):
            if X[i].shape[0] != pad_value:
                X[i] = np.concatenate((X[i] , np.ones((pad_value - X[i].shape[0], X[i].shape[1])) * 255), axis=0)

        X = tf.keras.preprocessing.sequence.pad_sequences(X, value= float(255),dtype="float32", padding="post")
        y["notes_labels"] = pad_sequences(y["notes_labels"],
                                          value=self.n_classes,
                                          dtype="int32",
                                          padding="post"
                                         )
        y["octaves_labels"] = pad_sequences(y["octaves_labels"],
                                          value=self.n_classes,
                                          dtype="int32",
                                          padding="post"
                                         )
        y["rythms_labels"] = pad_sequences(y["rythms_labels"],
                                          value=self.n_classes,
                                          dtype="int32",
                                          padding="post"
                                         )
        
        # Store class 
        n,length, height = X.shape
        
        return [np.reshape(X, [n, length, height, 1]), y, X_len, y_len]


    def convert_into_number(self, y, list_label):

        t = list_label.split('{')[1]
        t = t.split('}')[0]
        t = t.split(',')
        res = list()
        for j in range(len(t)):
            t[j] = t[j].split("'")[1]
        for i in y:
            for j in range(len(t)):
                if i== t[j]:
                    res.append(j)
                    break
        return res


    def labels_for_image(self,f,imagename):
        s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        byte_number = s.find(imagename.encode('utf-8'))
        image_labels = f.read()[byte_number:].split('\n')[0]
        return (image_labels.split('|')[1:])


    def notes_label(self, f, imagename):
        image_labels = self.labels_for_image(f, imagename)
        for res in image_labels[2:] :
            temp = res.split('_')
            if temp[0] in CLASS_NAMES_NOTES:
                yield temp[0]
            elif temp[0] == '':
                yield '|'


    def octaves_label(self, f, imagename):
        image_labels = self.labels_for_image(f,imagename)
        for res in image_labels[2:]:
            temp = res.split('_')
            if len(temp) > 2:
                yield temp[1]
            elif temp[0] == '':
                yield '|'
            elif temp[0] == 'rest' :
                yield temp[0]

    def rythms_label(self,f, imagename):
        image_labels = self.labels_for_image(f,imagename)
        
        for res in image_labels[2:]:
            temp = res.split('_')
            if len(temp) > 1:
                yield(temp[-1])
            elif temp[0] == '':
                yield '|'


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

