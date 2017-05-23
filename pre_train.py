import sysconfig
sysconfig.get_config_var('Py_UNICODE_SIZE')
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

for filename in ["my_gap_ResNet50_1.h5", "my_gap_InceptionV3_1.h5"]: #, "my_gap_Xception.h5"
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['mytrain1']))
        X_test.append(np.array(h['mytest1']))
        y_train = np.array(h['mylabel1'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)
y_train = keras.utils.to_categorical(y_train, num_classes)

print X_train.shape
print y_train.shape
print X_test.shape

X_train, y_train = shuffle(X_train, y_train)

input_tensor = Input(X_train.shape[1:])
x = Dropout(0.5)(input_tensor)
x = Dense(num_classes, activation='softmax')(x)
model = Model(input_tensor, x)

model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=10, nb_epoch=100, validation_split=0.2)

# print ("Building model......")
# model = Sequential()
# model.add(Dense(2048, input_shape=X_train.shape[1:]))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# # model.add(Dense(64))
# # model.add(Activation('relu'))
# # model.add(Dropout(0.75))
# model.add(Dense(num_classes))
# model.add(Activation("softmax"))
# model.compile(optimizer='adadelta',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# history = model.fit(X_train, y_train,
#                     nb_epoch=20, batch_size=16,
#                     verbose=1, validation_split=0.2)

# print history

