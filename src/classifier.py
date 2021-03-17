"""
File containing the actual neural network (or custom class, etc.) that does the
image classification
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class MathDigitModel(tf.keras.Model):

    def __init__(self, number_layers = 1, neurons_per_layer = 100):

        """
        Initialization of the MathDigitModel instances

        Params
        ------
        * number_layers is the number of layers in the neural network
        * neurons_per_layer is the number of neurons per layer - tbd what
        the best value is because too low and the classifier will be poor, but too high
        and the classifier will take too long to train
        """
        super(MathDigitModel, self).__init__()
        self.layers = number_layers
        self.neurons = neurons_per_layer

        # need to have at least one layer
        if self.layers < 1:
            raise Exception("number of layers must be > 1")

        # store layers in list so that we can add as many layers as we want
        self.dense = []
        for i in range(number_layers):
            self.dense.append(tf.keras.layers.Dense(neurons_per_layer, activation=tf.nn.relu))
        self.dense.append(tf.keras.layers.Dense(neurons_per_layer, activation=tf.nn.softmax))
        # reserves half the data for training
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=False):
        """
        This method is used to actually perform "classification"; i.e., given
        some input data (possibly stacked), return a probability distribution
        (i.e., the final layer of the network) that specifies how to classify
        the image input.

        Params
        ------
        inputs: (tf.tensor ? TODO include more details about type; capital T?)
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
        inputs.shape == (1, 2025) TODO: figure out - image files are 45x45
        """
        #feeds input data through first layer
        output = self.dense[0](inputs)

        #putting output of first layer into subsequent layer and so on
        if training:
            output = self.dropout(output, training=training)
        for i in range(1, self.layers):
            output = self.dense[i](output)

        #return last layer - the final output
        return output

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
        #compile the model
        self.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        #train the model
        # should we include other params in train fx like number of epochs ?
        self.fit(x=training_inputs, y=training_classes, epochs=5)
