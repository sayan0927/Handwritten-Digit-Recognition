# -*- coding: utf-8 -*-
"""
# Machine Learning CS404D Project

# Handwritten Digits Classification

# Submitted by
# Sayan Paul -          M210653CA , sayan_m210653ca@nitc.ac.in
# Chirangee Lal Verma - M210696CA , chirangee_m210696ca@nitc.ac.in 

# Problem Statement

# Digit recognition is a well
# known task in the field of computer vision, and it
# has numerous applications. The MNIST dataset is a popular benchmark
# dataset for this task, but it is often challenging to apply models trained on
# MNIST to images captured by us. Therefore, the problem statement for this
# project is to train a Convolutional Neural Network (CNN) on the MNIST
# dataset with some augmentations, to predict digits in custom
# images.The
# project aims to explore how well a model trained on the MNIST dataset can
# generalize to new images.



"""
from ModelLoaderClass import ModelLoader

from PredictorClass import Predictor



test_images_path = "test\\"





model,x_test,y_test=ModelLoader('my_model.h5').get_model()





predictor = Predictor(model)
predictor.predict_on_test_data(x_test,y_test)
predictor.predict_on_all_images(test_images_path)







