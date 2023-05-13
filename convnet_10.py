"""
This is an example using MNIST dataset with our own customized MLP
"""

import tensorflow as tf
import numpy as np
import metrics.metrics as metrics
import keras.datasets as datasets
import os
import skimage.io as io
import convnet.simple as simple
import matplotlib.pyplot as plt
import skimage as ski

dir_path_10 = "QuickDraw-10"
dir_path_train_10 = "QuickDraw-10/train.txt"
dir_path_test_10 = "QuickDraw-10/test.txt"


images1 =[]
target1 =[]
images2 =[]
target2 =[]

def load_save_data(input_name, output_name):
    y=[]
    files =[]
    with open(input_name) as f:
        for x in f:
            fila = x.split("\t")
            y.append(int(fila[1]))
            files.append(fila[0])
    
    x = []
    for i, f in enumerate(files) :
        filename = os.path.join(dir_path_10, f)
        image = io.imread(filename)
        resized_image = ski.transform.resize(image, (128, 128))
       # fd = hog(image, orientations=8, pixels_per_cell=(32,32),
              #   cells_per_block=(1, 1), visualize = False)
        x.append(resized_image)
       # if i % 100 ==0:
           # print(i, flush = True)
    
    x = np.array(x)
    y = np.array(y)
    np.save(output_name + "_x.pny",x)
    np.save(output_name + "_y.pny",y)

load_save_data(dir_path_train_10, "train_10")
load_save_data(dir_path_test_10, "test_10")

x_train = np.load("train_10_x.pny.npy")
y_train= np.load("train_10_y.pny.npy")
x_test= np.load("test_10_x.pny.npy")
y_test= np.load("test_10_y.pny.npy")

def visualize_classes():
    for i in range(0, 10):
        img_batch = x_train[y_train == i][0:10]
        img_batch = np.reshape(img_batch, (img_batch.shape[0]*img_batch.shape[1], img_batch.shape[2]))
        if i > 0:
            img = np.concatenate([img, img_batch], axis = 1)
        else:
            img = img_batch
    plt.figure(figsize=(10,20))
    plt.axis('off')
    plt.imshow(img, cmap="gray")
    
# visualize_classes()

print ('{} {}'.format(x_train.shape, x_train.dtype))
print ('{} {}'.format(x_test.shape, x_train.dtype))
print ('{} {}'.format(y_train.shape, y_train.dtype))
print ('{} {}'.format(y_test.shape, y_train.dtype))

n_train = x_train.shape[0]
n_test = x_test.shape[0]

#reshape the images to represent  BxHxWxC
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)) 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

#converting labels to one-hot encoding
y_train_one_hot = tf.keras.utils.to_categorical(y_train)#to_one_hot(y_train)
y_test_one_hot = tf.keras.utils.to_categorical(y_test)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)


mu = np.mean(x_train, axis = 0)

# normalize data, just centering
x_train = (x_train - mu) 
x_test = (x_test - mu)  
np.save('mean.npy', mu)
# create the model
model = simple.SimpleModel(10)
model = model.model(x_train.shape[1:])    
model.summary()

# defining optimizer
opt = tf.keras.optimizers.SGD()
 
# put all together  
model.compile(
         optimizer=opt,              
          loss='categorical_crossentropy', 
          metrics=['accuracy'])
 
# training or fitting 
model.fit(x_train, 
        y_train_one_hot, 
        batch_size=256,  
        epochs = 10,
        validation_data = (x_test, y_test_one_hot))
 
# prediction using directly the trained model
# there is also a function called -- predict -- , you can check it  
y_pred = model(x_test, training = False)
print(y_test)
print(y_pred)
print(y_pred.shape)
# computing confusion_matrix
mc = metrics.confusion_matrix(y_test, y_pred, 10)
model_file = 'emnist_model'
model.save(model_file)
print('model was saved at {}'.format(model_file))
# print mc
print(mc)
# mc as percentages
rmc = mc.astype(np.float32) / np.sum(mc, axis = 1, keepdims = True)
rmc = (rmc * 100).astype(np.int32) / 100 
print(rmc)