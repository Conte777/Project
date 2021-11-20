import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

def create():
    model = keras.Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
        ])

    print(model.summary())      # вывод структуры НС в консоль

    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])


    model.fit(x_train,
              y_train_cat,
              batch_size=32,
              epochs=5,
              verbose=2,
              validation_split=0.2)

    model.evaluate(x_test, y_test_cat)

    model.save('model_chi')

def solving(n):
    model = keras.models.load_model('model_chi')
    x = np.expand_dims(x_test[n], axis=0)
    res = model.predict(x)
    print( res )
    print( np.argmax(res) )
    print( f"Распознаная цифра: {np.argmax(res)}" )
    print( f"Привильный ответ: {y_test[n]}" )

    plt.imshow(x_test[n], cmap=plt.cm.binary)
    plt.show()

if __name__ == "__main__":
    pass
##    create()
##    x = input()
##    while x != '':
##        solving(int(x))
##        x = input()
