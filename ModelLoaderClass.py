# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 17:21:04 2023

@author: sayan
"""

import os
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from DataLoaderClass import DataLoader



class ModelLoader:
    
    def __init__(self,model_name):
        self.model_name=model_name
        
        
    def get_model(self):
        
        
        # if model exists , just load and return it
        
        file_exists=os.path.isfile(self.model_name)
        data_loader=DataLoader()
        if file_exists:
            x_test=data_loader.get_x_test()
            y_test=data_loader.get_y_test()
            model=load_model(self.model_name)
            return model,x_test,y_test
        else:
            
            # load data 
            x_train=data_loader.get_x()
            y_train=data_loader.get_y()
            
            x_test=data_loader.get_x_test()
            y_test=data_loader.get_y_test()
            
            #create model
            
            model = Sequential()

            ## Declare the layers
            layer_1 = Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1))
            layer_2 = Conv2D(64, kernel_size=3, activation='relu')
            layer_3 = Flatten()
            layer_4 = Dense(10, activation='softmax')

            ## Add the layers to the model
            model.add(layer_1)
            model.add(layer_2)
            model.add(layer_3)
            model.add(layer_4)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
          
            model.fit(x_train,y_train, batch_size=(100),epochs=5)
            model.save(self.model_name)
            
            return model,x_test,y_test

        
        
        
        