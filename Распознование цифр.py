import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from PIL import Image

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)


def create():
    model = keras.Sequential([Conv2D(32, (3,3),
    padding='same',
    activation='relu',
    input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10,  activation='softmax')
    ])

    # print(model.summary())      # вывод структуры НС в консоль

    model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


    model.fit(x_train, y_train_cat, batch_size=32, epochs=5, verbose=2, validation_split=0.2)
    model.save('model_chi2')

def predicting(path_to_img):
    model = keras.models.load_model('model_chi2')
    img = Image.open(path_to_img)
    # изменение рзмера изобржений на 28x28
    img = img.resize((28,28))
    # конвертируем rgb в grayscale
    img = img.convert('L')
    img = np.array(img)
    # изменение размерности для поддержки модели ввода и нормализации
    img = img.reshape(1,28,28,1)
    img = img/255.0
    # предстказание цифры
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

if __name__ == "__main__":
    pass
    
