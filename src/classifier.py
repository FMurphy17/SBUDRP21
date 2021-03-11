"""
File containing the actual neural network (or custom class, etc.) that does the
image classification
"""

import tensorflow as tf


class MathDigitModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        """
        Initialization of the MathDigitModel instances

        Params
        ------
        [ write out each parameter, its type, and what it is for in English ]
        """
        super(MathDigitModel, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def call(self, inputs, training=False):
        """
        This method is used to actually perform "classification"; i.e., given
        some input data (possibly stacked), return a probability distribution
        (i.e., the final layer of the network) that specifies how to classify
        the image input.

        Params
        ------
        inputs: (tf.tensor ? TODO include more details about type)
            Possibly stacked inputs to be put through the classifier
        training: (bool)
            Whether or not the inputs are to be understood as training data

        Returns
        -------
        dist: (tf.tensor)
            The probability distribution over the possible classes based on our
            model.

        Preconditions
        -------------
        inputs.shape == () TODO: figure out
        """
        return 0.

    def train(self, training_inputs, training_classes):
        """
        Fit the model to the training data using keras optimization builtins.

        For example, we may want to minimize cross-entropy as a performance
        measure. This is built-in to keras (I'm almost positive).

        Params
        ------
        training_inputs: (tf.tensor)
            Collection of images to be trained on
        training_classes: (tf.tensor)
            Collection of associated classes for each image 

        Returns
        -------
        [N/A called for side-effects only]

        Preconditions
        -------------
        The ith training input corresponds to the ith training class
        """
        pass

