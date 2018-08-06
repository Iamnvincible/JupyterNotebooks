# Keras Keynotes
## imports

from keras.datasets import <cifar10>#dataset name
from keras.utils import np_utils# to_categorical
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten #layer features
from keras.layers.convolutional import Conv2D,MaxPooling2D #Conv features
from keras.optimizers import SGD,Adam,RMSprop #optimizers
import matplotlib.pyplot as plt #plot

## data processing
### one hot
Y_train = np_utils.to_categorical(y_train,NB_CLASSES)
### cast to float32
X_train = X_train.astype('float32')

## model building
model=Sequential()
### Conv
Conv2D(filters,(kernel size,),padding,input_shape)
model.add(Conv2D(32,(3,3),padding='same',input_shape=(row,col,channel)))
model.add(layer)
Activation('relu')#softmax
MaxPooling2D(pool_size=(2,2))
Dropout(0.25)
Flatten()
Dense()# normal layers

## after building
model.compile(loss='categorical_crossentropy',optimizer=OPTIM,metrics=['accuracy'])

history = model.fit(X_train,Y_train,batch_size=BATCH_SIZE,epochs=NB_EPOCH,validation_split=VALIDATION_SPLIT,verbose=VERBOSE)

score = model.evaluate(X_test,Y_test,batch_size=BATCH_SIZE,verbose=VERBOSE)
print('test score:',score[0])
print('test accuracy:',score[1])

## ploting
import matplotlib.pyplot as plt
%matplotlib inline
def training_vis(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']

    # make a figure
    fig = plt.figure(figsize=(12,4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    ax1.plot(val_loss,label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label='train_acc')
    ax2.plot(val_acc,label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
training_vis(history)

predictions=model.predict_classes(X_test)
import pandas as pd
pd.crosstab(y_test.reshape(-1),predictions,rownames=['label'],colnames=['predict'])

## model saving and loading