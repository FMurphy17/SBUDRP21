import processing as prc
import classifier as clf
import numpy as np

trainiter, validiter = prc.load_data(prc.DATA_PATH)


model = clf.MathDigitModel(final_layer_units=trainiter.num_classes, input_shape=trainiter.image_shape)


#### Model creation is now done and we use the model by compile/fit/evaluate

model.train(trainiter)

# Evaluate the model performance
test_loss, test_acc = model.evaluate(x=validiter)
# Print out the model accuracy
print('\nTest accuracy:', test_acc)

#predictions = model.predict(validiter) # Make prediction
#print(np.argmax(predictions[1000])) # Print out the number

