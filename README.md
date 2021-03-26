# SBUDRP21

Stony Brook University Directed Reading Program (Spring 2021) on machine learning

We want to build an image classifier that converts handwritten math text to
LaTeX.

# Timeline

# Things left to do...

* ~~Find datasets for training~~
* ~~Set up an actual classifier using Keras or Torch or whatever as the neural
  network basis~~
* Improve the classifier
  * What metric are we trying to optimize? Default: accuracy. Is this a good measure? Let's maybe do a review of the discussion on what measure to optimize.
  * How are we optimizing? Default: ADAM. Probably not a big issue, but there are hyperparameters to tune.
  * How do we get our classifier to train "decently"? Currently it's total garbage, spitting out results that are worse than guessing randomly. (This probably relates to the preprocessing step).
  * (David) Feature: pause, save, load, resume training after program termination.
  * Investigate the training "loop". It seems like just setting the `epoch` keyword does the trick. But we should investigate all of the settings to `fit` and `fit_generator`.
  * Reporting classification results.
    * Somehow we should be able to provide visualization / output data / something useful for people to be able to inspect and consider the success of the model. 
      * Some kind of visualization, such as image + classification + true value
      * Confusion matrix? $A[i][j]$ is the percent of class-$j$ tokens that are classified as $i$.
      * Ties back in with the measure of performance we are interested in.
      * Model summary info.
* Data preprocessing
  * Investigate best practices for image scaling, normalization, etc. 
  * 
* Find some existing models (in papers or in practice, like on Github) where
  people do this (for example, Detexify) and try to replicate them
* Maybe if there's time, do an interactive demo
  * Draw a symbol -> save -> classify.
  * Or maybe there's some way to do optical character recognition where the handwriting is recognized immediately through a program.
* Get up to speed on doing deep learning 

# Datasets
* https://www.kaggle.com/xainano/handwrittenmathsymbols
* https://www.kaggle.com/clarencezhao/handwritten-math-symbol-dataset

# Links

* [Towards Data Science on Python OCR from scratch](https://towardsdatascience.com/create-simple-optical-character-recognition-ocr-with-python-6d90adb82bb8)
* [Towards Data Science - Image pre-processing](https://towardsdatascience.com/image-pre-processing-c1aec0be3edf)
* https://keras.io/examples/vision/captcha_ocr/
* https://www.tensorflow.org/tutorials/keras/classification#set_up_the_layers

* Existing projects: 
  * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/training.py
  * https://github.com/Otman404/Mathematical_Symbols_Recognition/tree/master/Project
  * https://github.com/ThomasLech/CROHME_extractor
