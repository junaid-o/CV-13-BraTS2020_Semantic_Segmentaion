"""
Custom data generator to work with BraTS2020 dataset.
Can be used as a template to create your own custom data generators. 

No image processing operations are performed here, just load data from local directory
in batches. 

"""

#from tifffile import imsave, imread
import os
import numpy as np
from matplotlib import pyplot as plt
import random


class DataGenerator:

    def __init__(self):
        pass

    def load_img(self, img_dir, img_list):
        images=[]
        for i, image_name in enumerate(img_list):    
            if (image_name.split('.')[1] == 'npy'):
                
                image = np.load(img_dir+image_name)
                        
                images.append(image)
        images = np.array(images)
        
        return(images)


    def imageLoader(self, img_dir, img_list, mask_dir, mask_list, batch_size):

        L = len(img_list)

        #keras needs the generator infinite, so we will use while true  
        while True:

            batch_start = 0
            batch_end = batch_size

            while batch_start < L:
                limit = min(batch_end, L)
                        
                X = self.load_img(img_dir, img_list[batch_start:limit])
                Y = self.load_img(mask_dir, mask_list[batch_start:limit])

                yield (X,Y) #a tuple with two numpy arrays with batch_size samples     

                batch_start += batch_size   
                batch_end += batch_size

