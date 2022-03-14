#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('tensorflow_version', '2.x')


# In[ ]:


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D,BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
import collections


# In[ ]:


(x_train, y_train), (x_test, y_test)= tf.keras.datasets.cifar10.load_data()


# In[ ]:


labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

W_grid = 10
L_grid = 10

fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,17))

axes = axes.ravel() 
n_train = len(y_train) 

for i in np.arange(0, W_grid * L_grid):
    index = np.random.randint(0, n_train) 
    axes[i].imshow(x_train[index,1:])
    label_index = int(y_train[index])
    axes[i].set_title(labels[label_index], fontsize = 8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)


# In[ ]:


print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[ ]:


#Normalize the data
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train=x_train/255
x_test=x_test/255


# In[ ]:


# Convert class vectors to binary class matrices. This is called one hot encoding.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# In[ ]:


model=Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# CONV => RELU => CONV => RELU => POOL => DROPOUT
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# FLATTERN => DENSE => RELU => DROPOUT
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
# a softmax classifier
model.add(Dense(10))
model.add(Activation('softmax'))


# In[ ]:


model.summary()


# In[ ]:


# initiate Adam optimizer
opt = keras.optimizers.Adam()

# Let's train the model using Adam
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[ ]:


hist=model.fit(x_train,y_train,batch_size=50,epochs=10,validation_split=0.2,shuffle=True)


# In[ ]:


model1=Sequential()
model1.add(Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:]))
model1.add(Activation('relu'))
model1.add(Conv2D(32,(3,3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))

# CONV => RELU => CONV => RELU => POOL => DROPOUT
model1.add(Conv2D(64, (3, 3), padding='same'))
model1.add(Activation('relu'))
model1.add(Conv2D(64, (3, 3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

# FLATTERN => DENSE => RELU => DROPOUT
model1.add(Flatten())
model1.add(Dense(512))
model1.add(Activation('relu'))
# a softmax classifier
model1.add(Dense(10))
model1.add(Activation('softmax'))


# In[ ]:


# initiate Adam optimizer
opt = keras.optimizers.SGD()

# Let's train the model using Adam
model1.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[ ]:


hist1=model1.fit(x_train,y_train,batch_size=50,epochs=10,validation_split=0.2,shuffle=True)


# In[ ]:


prediction_score = model1.evaluate(x_test, y_test, verbose=0)

print('Test Loss and Test Accuracy', prediction_score)


# In[ ]:


model2=Sequential()
model2.add(Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:]))
model2.add(Activation('relu'))
model2.add(Conv2D(32,(3,3)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))

# CONV => RELU => CONV => RELU => POOL => DROPOUT
model2.add(Conv2D(64, (3, 3), padding='same'))
model2.add(Activation('relu'))
model2.add(Conv2D(64, (3, 3)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

# FLATTERN => DENSE => RELU => DROPOUT
model2.add(Flatten())
model2.add(Dense(512))
model2.add(Activation('relu'))
# a softmax classifier
model2.add(Dense(10))
model2.add(Activation('softmax'))


# In[ ]:


# initiate Adam optimizer
opt = keras.optimizers.Adam(learning_rate=0.01)

# Let's train the model using Adam
model2.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[ ]:


hist2=model.fit(x_train,y_train,batch_size=50,epochs=10,validation_split=0.2,shuffle=True)


# In[ ]:


prediction_score = model2.evaluate(x_test, y_test, verbose=0)
print('Test Loss and Test Accuracy', prediction_score)


# In[ ]:


model3=Sequential()
model3.add(Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:]))
model3.add(Activation('relu'))
model3.add(Conv2D(32,(3,3)))
model3.add(Activation('relu'))
model3.add(MaxPooling2D(pool_size=(2,2)))

# CONV => RELU => CONV => RELU => POOL => DROPOUT
model3.add(Conv2D(64, (3, 3), padding='same'))
model3.add(Activation('relu'))
model3.add(Conv2D(64, (3, 3)))
model3.add(Activation('relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))

# FLATTERN => DENSE => RELU => DROPOUT
model3.add(Flatten())
model3.add(Dense(512))
model3.add(Activation('relu'))
# a softmax classifier
model3.add(Dense(10))
model3.add(Activation('softmax'))


# In[ ]:


# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model3.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[ ]:


hist3=model3.fit(x_train,y_train,batch_size=50,epochs=10,validation_split=0.2,shuffle=True)


# In[ ]:


prediction_score = model3.evaluate(x_test, y_test, verbose=0)
print('Test Loss and Test Accuracy', prediction_score)


# In[ ]:


def plotmodelhistory(history): 
    fig, axs = plt.subplots(1,2,figsize=(15,5)) 
    # summarize history for accuracy
    axs[0].plot(history.history['accuracy']) 
    axs[0].plot(history.history['val_accuracy']) 
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy') 
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'validate'], loc='upper left')
    # summarize history for loss
    axs[1].plot(history.history['loss']) 
    axs[1].plot(history.history['val_loss']) 
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss') 
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'validate'], loc='upper left')
    plt.show()


# In[ ]:


plotmodelhistory(hist)


# In[ ]:


plotmodelhistory(hist1)


# In[ ]:


plotmodelhistory(hist2)


# In[ ]:


plotmodelhistory(hist3)


# In[ ]:


#define the convnet
model4 = Sequential()
# CONV => RELU => CONV => RELU => POOL => DROPOUT
model4.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model4.add(Activation('relu'))
model4.add(Conv2D(32, (3, 3)))
model4.add(Activation('relu'))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(Dropout(0.25))

# CONV => RELU => CONV => RELU => POOL => DROPOUT
model4.add(Conv2D(64, (3, 3), padding='same'))
model4.add(Activation('relu'))
model4.add(Conv2D(64, (3, 3)))
model4.add(Activation('relu'))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(Dropout(0.25))

# FLATTERN => DENSE => RELU => DROPOUT
model4.add(Flatten())
model4.add(Dense(512))
model4.add(Activation('relu'))
model4.add(Dropout(0.5))
# a softmax classifier
model4.add(Dense(10))
model4.add(Activation('softmax'))

model4.summary()


# In[ ]:


# initiate Adam optimizer
opt = keras.optimizers.Adam(learning_rate=0.001)

# Let's train the model using Adam
model4.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[ ]:


hist4=model4.fit(x_train,y_train,batch_size=50,epochs=10,validation_split=0.2,shuffle=True)


# In[ ]:


prediction_score = model4.evaluate(x_test, y_test, verbose=0)
print('Test Loss and Test Accuracy', prediction_score)


# In[ ]:


#define the convnet
model5 = Sequential()
# CONV => RELU => CONV => RELU => POOL => DROPOUT=>Batch Normalization
model5.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model5.add(Activation('relu'))
model5.add(Conv2D(32, (3, 3)))
model5.add(Activation('relu'))
model5.add(MaxPooling2D(pool_size=(2, 2)))
model5.add(BatchNormalization())
model5.add(Dropout(0.25))

# CONV => RELU => CONV => RELU => POOL => DROPOUT=>Batch Normalization
model5.add(Conv2D(64, (3, 3), padding='same'))
model5.add(Activation('relu'))
model5.add(Conv2D(64, (3, 3)))
model5.add(Activation('relu'))
model5.add(MaxPooling2D(pool_size=(2, 2)))
model5.add(BatchNormalization())
model5.add(Dropout(0.25))

# FLATTERN => DENSE => RELU => DROPOUT=>Batch Normalization
model5.add(Flatten())
model5.add(Dense(512))
model5.add(Activation('relu'))
model5.add(BatchNormalization())
model5.add(Dropout(0.5))
# a softmax classifier
model5.add(Dense(10))
model5.add(Activation('softmax'))

model5.summary()


# In[ ]:


# initiate Adam optimizer
opt = keras.optimizers.Adam(learning_rate=0.001)

# Let's train the model using Adam
model5.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[ ]:


hist5=model5.fit(x_train,y_train,batch_size=50,epochs=10,validation_split=0.2,shuffle=True)


# In[ ]:


prediction_score = model5.evaluate(x_test, y_test, verbose=0)
print('Test Loss and Test Accuracy', prediction_score)


# In[ ]:


plotmodelhistory(hist5)


# In[ ]:


#define the convnet
model6 = Sequential()
# CONV => RELU => CONV => RELU => POOL => DROPOUT=>Batch Normalization
model6.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model6.add(Activation('selu'))
model6.add(Conv2D(32, (3, 3)))
model6.add(Activation('relu'))
model6.add(MaxPooling2D(pool_size=(2, 2)))
model6.add(BatchNormalization())
model6.add(Dropout(0.25))

# CONV => RELU => CONV => RELU => POOL => DROPOUT=>Batch Normalization
model6.add(Conv2D(64, (3, 3), padding='same'))
model6.add(Activation('selu'))
model6.add(Conv2D(64, (3, 3)))
model6.add(Activation('relu'))
model6.add(MaxPooling2D(pool_size=(2, 2)))
model6.add(BatchNormalization())
model6.add(Dropout(0.25))

# FLATTERN => DENSE => RELU => DROPOUT=>Batch Normalization
model6.add(Flatten())
model6.add(Dense(512))
model6.add(Activation('selu'))
model6.add(BatchNormalization())
model6.add(Dropout(0.5))
# a softmax classifier
model6.add(Dense(10))
model6.add(Activation('softmax'))


# In[ ]:


# initiate Adam optimizer
opt = keras.optimizers.Adam(learning_rate=0.001)

# Let's train the model using Adam
model6.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[ ]:


hist6=model6.fit(x_train,y_train,batch_size=50,epochs=10,validation_split=0.2,shuffle=True)


# In[ ]:


prediction_score = model6.evaluate(x_test, y_test, verbose=0)
print('Test Loss and Test Accuracy', prediction_score)


# In[ ]:




