"""
File containing the actual neural network (or custom class, etc.) that does the
image classification
"""


import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt


class MathDigitModel(tf.keras.Model):

    def __init__(self, number_layers=2, neurons_per_layer=128, final_layer_units=10, input_shape=None):

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
        self.n_layers = number_layers
        self.n_neurons = neurons_per_layer

        # need to have at least one layer
        if self.n_layers < 1:
            raise Exception("number of layers must be > 1")

        # store layers in list so that we can add as many layers as we want
        self.dense = []

        # these layers came from an example here:
        # https://machinelearningmastery.com/how-to-normalize-center-and-standardize-images-with-the-imagedatagenerator-in-keras/
        self.dense.append(tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=input_shape))
        self.dense.append(tf.keras.layers.MaxPooling2D((2, 2)))
        self.dense.append(tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu))
        self.dense.append(tf.keras.layers.MaxPooling2D((2, 2)))


        self.dense.append(tf.keras.layers.Flatten())
        for i in range(self.n_layers):
            self.dense.append(tf.keras.layers.Dense(self.n_neurons, activation=tf.nn.relu))
        self.dense.append(tf.keras.layers.Dense(final_layer_units, activation=tf.nn.softmax))
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
        output = inputs

        #putting output of first layer into subsequent layer and so on
        if training:
            output = self.dropout(output, training=training)
        for i in range(len(self.dense)):
            output = self.dense[i](output)

        #return last layer - the final output
        return output

    def train(self, training_inputs, training_classes=None):
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
        # https://datascience.stackexchange.com/questions/41921/sparse-categorical-crossentropy-vs-categorical-crossentropy-keras-accuracy
        #self.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        #train the model
        if training_classes is None:
            self.fit(training_inputs, epochs=5)
        else:
            self.fit(x=training_inputs, y=training_classes, epochs=5)

    def train_generator(self, training_iterator, validation_iterator):
        """
        Fit the model to the generator training data using keras optimization
        builtins.

        For example, we may want to minimize cross-entropy as a performance
        measure. This is built-in to keras (I'm almost positive).

        Params
        ------
        training_iterator: DataIterator
            training set lazily loaded to memory
        validation_iterator: DataIterator
            validation set lazily loaded to memory

        Returns
        -------
        [N/A called for side-effects only]
        """

        self.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.fit_generator(
            training_iterator, 
            validation_data=validation_iterator,
            steps_per_epoch=16,
            validation_steps=8,
        )

