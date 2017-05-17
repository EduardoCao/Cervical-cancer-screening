import h5py
import numpy as np
from sklearn.utils import shuffle
import keras.utils

from keras.models import *
from keras.layers import *

np.random.seed(2017)

X_train = []
X_test = []
num_classes = 3

for filename in ["my_gap_ResNet50.h5", "my_gap_InceptionV3.h5", "my_gap_Xception.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['mytrain0']))
        X_test.append(np.array(h['mytest0']))
        y_train = np.array(h['mylabel0'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)
y_train = keras.utils.to_categorical(y_train, num_classes)

X_train, y_train = shuffle(X_train, y_train)

input_tensor = Input(X_train.shape[1:])
x = Dropout(0.5)(input_tensor)
x = Dense(num_classes, activation='softmax')(x)
model = Model(input_tensor, x)

model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=10, nb_epoch=100, validation_split=0.2)

