import numpy as np
import tensorflow as tf
from tensorflow import keras


def norm(x):
    return (x - np.mean(x)) / np.std(x)

# prepare data
rows = 1000
test_ratio = 0.2
variables = 2
W, b = [5, 8], 6
X = np.random.randint(1, 1000, rows * variables) / 100
X = X.reshape(rows, variables)
X = norm(X)
y = np.dot(X, W) + b + np.random.random(rows)
data = np.c_[X, y]
np.random.shuffle(data)

train_data = data[:int(rows * (1 - test_ratio)), :variables]
train_labels = data[:int(rows * (1 - test_ratio)), variables]
test_data = data[int(rows * (1 - test_ratio)):, :variables]
test_labels = data[int(rows * (1 - test_ratio)):, variables]


def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=[train_data.shape[1]]),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

model = build_model()

model.summary()


model.fit(train_data,
          train_labels,
          epochs=20,
          batch_size=64,
          validation_data=(test_data, test_labels),
          verbose=2)


results = model.evaluate(test_data, test_labels, verbose=2)

print(results)

print(np.argmax(model.predict(test_data), axis=1))
print(test_labels)
