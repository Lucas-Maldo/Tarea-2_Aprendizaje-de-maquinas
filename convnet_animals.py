"""
This is an example using MNIST dataset with our own customized MLP
"""

import tensorflow as tf
import numpy as np
import metrics.metrics as metrics
import keras.datasets as datasets
import os
import skimage.io as io
from skimage.feature import hog
import convnet.simple as simple
import matplotlib.pyplot as plt
import skimage as ski
from PIL import Image

# dir_path_animals ="QuickDraw-Animals"
dir_path_train_animals = "QuickDraw-Animals/train_images"
dir_path_test_animals = "QuickDraw-Animals/test_images"

def load_save_data(input_name):
    y=[]
    files =[]
    with open(input_name) as f:
        for x in f:
            fila = x.split("\t")
            files.append(fila[0])
    return files

# loading dataset
print(load_save_data("QuickDraw-Animals/mapping.txt"))
etiquetas = load_save_data("QuickDraw-Animals/mapping.txt")

label_to_idx = {label: i for i, label in enumerate(etiquetas)}
print(label_to_idx)

def load_save_data(dir_path, etiquetas, output_name):
    images =[]
    target =[]
    for label in etiquetas:
        label_path = os.path.join(dir_path, label)
        for image_path in os.listdir(label_path):
                image = Image.open(os.path.join(label_path, image_path))
                image_array = np.array(image)
                image_array = ski.transform.resize(image_array,[128,128])/255
                images.append(image_array)
                target.append(label_to_idx[label])

    x = np.array(images)
    y = np.array(target)
    np.save(output_name + "_x.pny",x)
    np.save(output_name + "_y.pny",y)

load_save_data(dir_path_train_animals, etiquetas, "train_animals")
load_save_data(dir_path_test_animals, etiquetas, "test_animals")

x_train = np.load("train_animals_x.pny.npy")
y_train= np.load("train_animals_y.pny.npy")
# randomize = np.arange(len(x_train))
# np.random.shuffle(randomize)
# x_train = x_train[randomize]
# y_train = y_train[randomize]
# x_train, X_sparse, y_train = shuffle(x_train, coo_matrix(x_train), y_train, random_state=0)

x_test= np.load("test_animals_x.pny.npy")
y_test= np.load("test_animals_y.pny.npy")
# randomize = np.arange(len(x_test))
# np.random.shuffle(randomize)
# x_test = x_test[randomize]
# y_test = y_test[randomize]
# x_test, X_sparse, y_test = shuffle(x_test, coo_matrix(x_test), y_test, random_state=0)


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
model = simple.SimpleModel(12)
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
print(metrics.multiclass_accuracy(y_test, y_pred))

# computing confusion_matrix
mc = metrics.confusion_matrix(y_test, y_pred, 12)
valores = [0,1,2,3,4,5,6,7,8,9,10,11]
promedios = []
for i in range(len(mc)):
    promedios.append(max(mc[i])/sum(mc[i]))
print(promedios)
plt.title("Accuracy per class")
plt.bar(valores, promedios)
plt.show()
model_file = 'emnist_model'
model.save(model_file)
print('model was saved at {}'.format(model_file))
# print mc
print(mc)
# mc as percentages
rmc = mc.astype(np.float32) / np.sum(mc, axis = 1, keepdims = True)
rmc = (rmc * 100).astype(np.int32) / 100 
print(rmc)