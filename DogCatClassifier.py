import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import os

imgtargetshape=(150,150)


#organize data into cat, dog categories
trainfoldername="DogCatClassifier Training Data"
filenamelist=os.listdir(trainfoldername)
categories=[]
for i in filenamelist:
    if i.split('.')[0]=='cat':
        categories.append(0)
    else:
        categories.append(1)

datafr=pd.DataFrame({
    'Filename':filenamelist,
    'Category':categories
})

#CNN model

model=keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150,150,3)),

    keras.layers.Conv2D(32,(3,3),activation='relu'),
    keras.layers.Dropout(0.25),

    keras.layers.Flatten(),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(2,activation='softmax')
])

optimfunc=Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy",optimizer=optimfunc,metrics=['accuracy'])

datafr["Category"]=datafr['Category'].replace({0:'cat',1:'dog'})

trainset,validset=train_test_split(datafr,test_size=0.1111)
trainset=trainset.reset_index(drop=True)
validset=validset.reset_index(drop=True)

trainnum=trainset.shape[0]
validnum=validset.shape[0]

#training set
trainaugment=ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    rescale=0.1
)
batchsize=32
traingen=trainaugment.flow_from_dataframe(
    trainset,
    trainfoldername,
    x_col='Filename',
    y_col='Category',
    target_size=imgtargetshape,
    batch_size=batchsize
)

#validation set
validaugment=ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    rescale=0.1
)
validgen=validaugment.flow_from_dataframe(
    validset,
    trainfoldername,
    x_col='Filename',
    y_col='Category',
    target_size=imgtargetshape,
    batch_size=batchsize
)

history=model.fit(
    traingen,
    epochs=20,
    validation_steps=validnum//batchsize,
    steps_per_epoch=trainnum//batchsize,
    validation_data=validgen,
)