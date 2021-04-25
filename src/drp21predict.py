import os
import sys
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import processing as prc
import numpy as np

# load the labels. these are just the names of the folders in 
# the extracted_images folder

labels = sorted([os.path.basename(i) for i in glob.glob("../data/extracted_images/*")])

# load our saved model that we trained

model = keras.models.load_model('./testmodel.data')

# for each image file given on the commandline:
#   1. load the image
#   2. preprocess is to make it white-on-black
#   3. ask the classifier to predict a label
# see: https://keras.io/api/preprocessing/image/

for fn in sys.argv[1:]:
    image = keras.preprocessing.image.load_img(fn, color_mode="grayscale")
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    input_arr = np.abs(input_arr-255.0)
    p1 = model.predict(input_arr)

    maxpred = p1.max()
    predindex = int(p1.argmax(axis=-1))
    predlabel = labels[predindex]
    print(f"{os.path.basename(fn)} is predicted to be '{predlabel}' with probabilty: {maxpred:.3f}")
