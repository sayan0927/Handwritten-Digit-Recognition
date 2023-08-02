# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 21:02:46 2023

@author: sayan
"""
from scipy import ndimage
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

class ImagePreProcessor:
    
    def __init__(self,image_path,image_name):
        
        
        self.image_path=image_path
        self.image_name=image_name
        self.preprocessed_digits=[]
        self.process_digits(self.image_path,self.image_name)
        
        
        
        
        
        
    def get_preprocessed_digits(self):
        return self.preprocessed_digits
        
        
   
    def getBestShift(self,img):
        #find center  of the digit
        cy,cx = ndimage.measurements.center_of_mass(img)
        rows,cols = img.shape
        
        # (cols/2.0-cx) gives distance between centre of image and centre of digit
        
        # find the shift on x and y to centre the digit  (shift is the distance we found)
        dx = np.round(cols/2.0-cx).astype(int)
        dy = np.round(rows/2.0-cy).astype(int)
        return dx,dy



    def shift(self,img,dx,dy):
        rows,cols = img.shape
        
        
        # a point in the image is (x,y) we can write it as (x,y,1)
        
        
        #                     |1  0  dx |   | x |
        #                     |0  1  dy | * | y |  = ( x +dx , y + dy )
        #                     |0  0   1 |   | 1 |
        
        # create transform matrix 
        M = np.float32([[1,0,dx],[0,1,dy]])
        #apply transform
        shifted = cv2.warpAffine(img,M,(cols,rows))
        return shifted
    
    def process_digits(self,image_path,image_name):
     
        image = cv2.imread(image_path+image_name)
        grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        
        plt.imshow(grey,cmap='gray')
        plt.title("digits in "+image_name, )
        plt.show()
        
        
        
        # invert colors , pixels with values  < 130 set to 255 and values > 130 set to 0
        ret, thresh = cv2.threshold(grey.copy(), 130, 255, cv2.THRESH_BINARY_INV)
        
        
        plt.imshow(thresh,cmap='gray')
        plt.show()
        
        #identify the contours in the inverted image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("digits in "+image_name, )
        plt.show()
        
        # sort contours based on x axis values to get the digits in sequential order
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        for c in contours:
            
            #get coordinates of top left corner (x,y) and width , height of the contour
            x,y,w,h = cv2.boundingRect(c)
            if w < 50 and h < 50:
                continue
            
            #cropping out the digit
            digit = thresh[y:y+h, x:x+w]
            # Resizing that digit 
            resized_digit = cv2.resize(digit, (28,28))
            gray=resized_digit
            
            #removing any row or columns that are completely black
            while np.sum(gray[0]) == 0:
                gray = gray[1:]

            while np.sum(gray[:,0]) == 0:
                gray = np.delete(gray,0,1)

            while np.sum(gray[-1]) == 0:
                    gray = gray[:-1]
                    
            while np.sum(gray[:,-1]) == 0:
                gray = np.delete(gray,-1,1)



            rows,cols = gray.shape
            
            # scale the images down while preserving aspect ration
            if rows > cols:
                factor = 20.0/rows
                rows = 20
                cols = int(round(cols*factor))
                
                gray = cv2.resize(gray, (cols,rows))
            else:
                factor = 20.0/cols
                cols = 20
                rows = int(round(rows*factor))
                gray = cv2.resize(gray, (cols, rows))
                
            # now images are 20 x n or n x 20 where n<=20
            
            # add padding to top,bottom,left,right
            
            colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
            rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
            gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
            gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
            
            # shift the centre of mass of digit to centre of image
            dx,dy = self.getBestShift(gray)
            shifted = self.shift(gray,dx,dy)
            
            # finally resize the digit to 28 x 28
            gray = shifted
            gray=cv2.resize(gray,(28,28))
            padded_digit=gray
            
            # normalise the digit , this is final step in digit processing
            padded_digit=padded_digit/255
            plt.imshow(padded_digit,cmap='gray')
            plt.show()
           
            
            self.preprocessed_digits.append(padded_digit)
        
        return