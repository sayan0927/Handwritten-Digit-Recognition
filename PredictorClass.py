# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 21:23:27 2023

@author: sayan
"""



import numpy as np

from ImagePreProcessorClass import ImagePreProcessor
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support

class Predictor:
    
    def __init__(self,model):
        self.model=model
        return
        
    
    def predict_on_all_images(self,image_path):
       
        files=os.listdir(image_path)
        for filename in files:
            self.predict_on_image(image_path,filename)
        return
    
    def predict_on_test_data(self,x_test,y_test):
         y_pred = self.model.predict(x_test)

         y_pred_classes = np.argmax(y_pred, axis=1)

         
         conf_mat = confusion_matrix(y_test, y_pred_classes)


         print('\n\nConfusion Matrix\n',conf_mat)
         precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_classes, average='weighted')
         print("\nPrecision:", precision,end='\n')
         print("\nRecall:", recall,end='\n')
         print("\nF1-score:", fscore,end='\n')
         return
        
    
    def predict_on_image(self,image_path,image_name):
      
     
       
            
        img_processor=ImagePreProcessor(image_path,image_name)
        preprocessed_digits=img_processor.get_preprocessed_digits()
       
        model = self.model
        predicted_digits=[]

        for digit in preprocessed_digits:    
            prediction = model.predict(digit.reshape(1, 28, 28, 1),verbose=0)
            predicted_digits.append(np.argmax(prediction))
            
        print("\nIn Image "+image_name, " Detected Digits")
        print(predicted_digits)
        # print(self.model.summary())
        return
    
    

