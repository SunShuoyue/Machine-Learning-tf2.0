import numpy as np
from tensorflow import keras


def norm(x):
    return (x - np.mean(x)) / np.std(x)


# prepare data
rows = 1000
test_ratio = 0.2
variables = 2
x1 = np.random.randint(900, 1000, [int(rows / 2), variables]) * 0.01
y1 = np.ones([int(rows / 2)])
data1 = np.c_[x1, y1]
x0 = np.random.randint(100, 200, [int(rows / 2), variables]) * 0.01
y0 = np.zeros([int(rows / 2)])
data0 = np.c_[x0, y0]
data = np.r_[data1, data0]
np.random.shuffle(data)

train_data = data[:int(rows * (1 - test_ratio)), :variables]
train_labels = data[:int(rows * (1 - test_ratio)), variables]
test_data = data[int(rows * (1 - test_ratio)):, :variables]
test_labels = data[int(rows * (1 - test_ratio)):, variables]


# train
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=[train_data.shape[1]]),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


model = build_model()
model.summary()
model.fit(train_data,
          train_labels,
          epochs=20,
          batch_size=64,
          validation_data=(test_data, test_labels),
          verbose=2)

# test
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
