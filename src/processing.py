import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATA_PATH = '../data/extracted_images/'


def load_data(path, validation_split=0.1, **kwargs):
    """
    Loads the handwritten mathematical symbol database and returns it as a
    generator. 

    The Python generator is a lazily-evaluated iteratable data structure. It
    allows the computer to process memory-intensive data without loading the
    entirety of the data onto the evaluation stack. 

    Tensorflow and Keras allow us to train directly on generator datasets. This
    is actually super cool, and will save us for this project.

    Parameters
    ----------
    path: str
        relative location of the data directory containing all of the training
        images and their classes.
    validation_split: float
        fraction of the training data to be reserved for validation
    class_mode: str (default: 'categorical')
        one of "categorical", "binary", "sparse", "input", or None. determines
        the type of label arrays that are returned: - "categorical" will be 2d
        one-hot encoded labels, - "binary" will be 1D binary labels, "sparse"
        will be 1d integer labels, - "input" will be images identical to input
        images (mainly used to work with autoencoders). - if None, no labels are
        returned (the generator will only yield batches of image data, which is
        useful to use with model.predict()). please note that in case of
        class_mode None, the data still needs to reside in a subdirectory of
        directory for it to work correctly.
    batch_size: int (default: 64)
        size of the batches of data.
    seed: int (default: 42)
        optional random seed for shuffling and transformations.
    shuffle: bool (default: True)
        whether to shuffle the data. if set to False, sorts the data in
        alphanumeric order.
    target_size: (int, int) (default: (256, 256))
        uple of integers (height, width). the dimensions to which all images
        found will be resized.
    color_mode: str (default: 'grayscale')
        one of "grayscale", "rgb", "rgba". whether the images will be converted
        to have 1, 3, or 4 channels.

    Returns
    -------
    train_iter, valid_iter: (DirectoryIterator, DirectoryIterator)
        iterators for the training and validation datasets, respectively
    """
    datagen = ImageDataGenerator(validation_split=validation_split)

    flow_kwargs = {
        'class_mode': 'categorical',
        'batch_size': 64,
        'seed': 42,
        'shuffle': True,
        'target_size': (256, 256),
        'color_mode': 'grayscale',
    }
    flow_kwargs.update(kwargs)

    train_it = datagen.flow_from_directory(path, subset='training', **flow_kwargs)
    validation_it = datagen.flow_from_directory(path, subset='validation',**flow_kwargs)

    return train_it, validation_it

