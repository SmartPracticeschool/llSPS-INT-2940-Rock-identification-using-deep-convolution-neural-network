# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:20:56 2020

@author: Prathmesh
"""


import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
#import tensorflow as tf
global graph
#graph = tf.compat.v1.get_default_graph()
from flask import Flask , request, render_template
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

app = Flask(__name__)
model = load_model("trained.h5")

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        #with graph.as_default():
        preds = model.predict_classes(x)
            
        #print("prediction",preds)
            
        labels = ['Conglomerate','Diorate','Fire opal','Genisis','Limestone','Obsidian','Slate']
        
        text = "the predicted rock is : " + str(labels[preds[0]])
        
    return text
if __name__ == '__main__':
    app.run(debug = False, threaded = False)
        
        
        
    
    
    
