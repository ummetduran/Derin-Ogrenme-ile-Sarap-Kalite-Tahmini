from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPool2D
from tensorflow.python.keras.layers import ZeroPadding2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Activation, \
    GlobalAveragePooling2D, Conv2D
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import Adadelta
from tensorflow.python.layers.core import Flatten
from keras.datasets import cifar100
from matplotlib import pyplot as plt
from keras.utils import to_categorical

from tensorflow_estimator.python.estimator import keras

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
img = plt.imshow(x_train[1])

y_train_ohe = to_categorical(y_train)
y_test_ohe = to_categorical(y_test)
# Ohe işleminden sonra y_train

x_train = x_train / 255
x_test = x_test / 255

# Model tanımlanır
model=Sequential()
#CNN ve pooling katmanları eklenir
model.add(Conv2D(32,(5,5),activation='relu',input_shape=(32,32,3)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,(5,5),activation='relu'))
model.add(Conv2D(32,(5,5),activation='relu'))
model.add(Conv2D(32,(5,5),activation='relu'))


model.add(MaxPool2D(pool_size=(2,2)))
#Flatten katmanı eklenir
#/model.add(Flatten())
model.add(Dense(1000,activation='relu') )
model.add(Dense(10,activation='softmax') )# 10 tane etiket olduğu için 10 tane sinir node uluşturuldu.

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# model.fit(x_train,y_train, validation_data=(x_test, y_test),
#                     batch_size=64,
#                     shuffle=False,
#                     verbose=1,
#                     epochs=12)

model.fit(x_train, y_train_ohe, batch_size=256, epochs=20, validation_split=0.3)

img, label = x_test[550]
plt.imshow(img.permute(1, 2, 0))
print('Label:', x_test.classes[label], ', Predicted:', model.predict(img, model))
