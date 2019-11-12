import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import pandas as pd
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

word_index = {k: (v + 3) for k, v in dictionary.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

classes = np.unique(df.label)
data = np.zeros([len(X), length + 1])
for i in range(len(X)):
    for j in range(len(X[i])):
        data[i][j] = word_index[X[i][j]]
        data[i][-1] = y[i]

np.random.shuffle(data)
test_ratio = 0.2
rows, variables = data[:, :-1].shape
train_data = data[:int(rows * (1 - test_ratio)), :variables]
train_labels = data[:int(rows * (1 - test_ratio)), variables]
test_data = data[int(rows * (1 - test_ratio)):, :variables]
test_labels = data[int(rows * (1 - test_ratio)):, variables]

# print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
# print(train_data[0])
# len(train_data[0]), len(train_data[1])
# def decode_review(text):
#     return ' '.join([reverse_word_index.get(i, '?') for i in text])
#
# decode_review(train_data[0])

train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=length)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=length)

vocab_size = len(word_index)

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 128))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(vocab_size, 64),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(3, activation='softmax')
# ])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_path = "model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1)

model.fit(train_data,
          train_labels,
          epochs=40,
          batch_size=64,
          validation_data=(test_data, test_labels),
          callbacks=[cp_callback],
          verbose=2)

model.load_weights(checkpoint_path)

results = model.evaluate(test_data, test_labels, verbose=2)

print(results)

print(np.argmax(model.predict(test_data), axis=1))
print(test_labels)
