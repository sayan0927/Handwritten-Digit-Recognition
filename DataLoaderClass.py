# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 20:40:12 2023

@author: sayan
"""

import os
import numpy as np
from keras.datasets import mnist
from ImagePreProcessorClass import ImagePreProcessor
import matplotlib.pyplot as plt
from PIL import Image
from keras.utils import to_categorical


import shutil

class DataLoader:
    
    
    def __init__(self):
        
        self.custom_image_path="train\\"
     
        
        
        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        self.x=x_train
        self.y=y_train
        
        self.x_test=x_test
        self.y_test=y_test
        
        
        #normalize
        
        self.x=self.x/255
        self.x_test=self.x_test/255
        
       
        self.check_for_custom_data()
        return 
       
        
       
       
       
       
    def get_x_test(self):
          np.save('x_test.npy', self.x_test)
          return self.x_test

    def get_y_test(self):
          np.save('y_test.npy', self.y_test)
          return self.y_test
       
    def get_x(self):
        np.save('x_train.npy', self.x)
        return self.x

    def get_y(self):
        np.save('y_train.npy', self.y)
        self.y = to_categorical(self.y, 10)
        return self.y
    
    def add_to_x(self,new):
        # add new datapoints to x (150 for each jpg file )
        self.x=np.concatenate((self.x,new))
        return
    
    
    def add_to_y(self):
        
        #add corresponding labels to y (150 for each jpg file)
        
        for i in range(0,10):
           vec=np.zeros((15,))
           vec.fill(i)
           self.y=np.concatenate((self.y,vec))
           
        return
    
    def check_for_custom_data(self):
       

        files=os.listdir(self.custom_image_path)
        self.digits=[]
        
        
          

        if not files:
           
            return
        else:
            
          # if  there are custom train images , process the images and add datapoints to x and y
           
           for file in files:
               name="train\\"+str(file)
               path="train\\"
               name=str(file)
               
               # process the file and d will hold 150 datapoints 
               d=self.process_image(path,name)
             
               self.add_to_x(d)
               self.add_to_y()
               
           
       
             
        return
        
           
               
    def process_image(self,path,name):
         
         preprocessor=ImagePreProcessor(path,name)
         
         # after pre processing the jpg file we get 10 digits in pre processed_digits scaled to 28 x 28
         
         preprocessed_digits=preprocessor.get_preprocessed_digits()
         
         digits=[]
         
         # rotate each of the 10 preprocessed_digits to create more data
         
         for digit in preprocessed_digits:
        
            rotated=self.get_rotated_images(digit)
            digits.extend(rotated)
                       
         return digits
             
             
             
    def get_rotated_images(self,img):
        
        rotated_images=[]
        img=Image.fromarray(img)
        
        #rotating 5 deg to 40 (anticlockwise)
        
        for degree in range(5,40,5):
            rotate_image=img.rotate(degree)
            rotate_image=np.array(rotate_image)
           
            rotated_images.append(rotate_image)
            
        #rotating -5 deg to -40 (clockwise)
            
        for degree in range(0,40,5):
            rotate_image=img.rotate(-degree)
            rotate_image=np.array(rotate_image)
            
            rotated_images.append(rotate_image)
            
        return rotated_images
    
    
    def clear_directory(self,path):
        directory_path = path

        # Delete the contents of the directory
        shutil.rmtree(directory_path)

        # Create an empty directory
        os.mkdir(directory_path)
        return
         
               
           