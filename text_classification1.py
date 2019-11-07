import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
import jieba

print(tf.__version__)

df = pd.read_csv('data/data.csv')

X = []
y = []
words = []
length = 0
for i in range(len(df)):
    word = jieba.lcut(df.iloc[i]['sentence'])
    X.append(word)
    y.append(df.iloc[i]['label'])
    length = max(length, len(word))
    words += word

dictionary = dict(zip(set(words), range(1, len(set(words)) + 1)))
classes = np.unique(df.label)
data = np.zeros([len(X), length + 1])
for i in range(len(X)):
    for j in range(len(X[i])):
        data[i][j] = dictionary[X[i][j]]
        data[i][-1] = y[i]

np.random.shuffle(data)
test_ratio = 0.2
rows, variables = data[:, :-1].shape
X_train = data[:int(rows * (1 - test_ratio)), :variables]
y_train = data[:int(rows * (1 - test_ratio)), variables]
X_test = data[int(rows * (1 - test_ratio)):, :variables]
y_test = data[int(rows * (1 - test_ratio)):, variables]

X_train = X_train / len(dictionary)
X_test = X_test / len(dictionary)

model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc)
