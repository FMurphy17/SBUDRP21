import processing as prc
import classifier as clf
import numpy as np

# Load the data

trainiter, validiter = prc.load_data(prc.DATA_PATH)

# Create the model object

model = clf.MathDigitModel(final_layer_units=trainiter.num_classes, input_shape=trainiter.image_shape)


# Train the model on our training set

model.train(trainiter)

# Evaluate the model performance using the validation set

test_loss, test_acc = model.evaluate(x=validiter)
print('\nTest accuracy:', test_acc)

# Save the model so we can do predictions and skip the training time

model.save('./testmodel.data')


